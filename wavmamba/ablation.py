"""Ablation study for WavMamba — a parallel, configurable assembler.

The shipped `model.WavMamba` is deliberately locked (it raises on any deviation
from the paper flags), so it cannot host the ablation variants. This module
provides `AblationWavMamba`, a parallel assembler that takes the SAME leaf
blocks from `model.py` (`SubbandStem`, `TFBlock`, `RMSNorm`, `AdaptiveFusion`,
`AttnStatPool`, `Classifier`) and composes them under real, swappable flags.
The paper model and its reproducibility guarantee are untouched.

Seven single-variable axes, each swapping exactly one component vs "ours"
(the paper configuration). See `ABLATIONS` for the registry.

    front_end : 'dwt'  (Haar HL+LH, ours)         | 'raw' (no DWT, single map)
    branch    : 'separate' (per-subband, ours)    | 'shared' (one branch)
    stem      : 'stem' (SubbandStem+3xTFBlock, ours) | 'nostem' (embed straight)
    backbone  : 'bimamba' (ours) | 'unimamba' | 'bilstm'
    merge     : 'gate' (per-channel zero-init, ours) | 'add' | 'concat'
    fusion    : 'adaptive' (softmax gate, ours) | 'mean' | 'concat'
    pool      : 'attnstat' (ECAPA, ours) | 'mean'

`AblationWavMamba(**ABLATIONS['ours']['kwargs'])` reproduces `WavMamba`
layer-for-layer (same parameter count); a test asserts this so the assembler
can never silently drift from the paper model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from .data import DATA_ROOT, DIRMAP
from .model import (
    HAS_MAMBA, _MAMBA_IMPORT_ERROR, _D_CONV, _DILATIONS, _DP_CNN, _DP_MAMBA,
    _EMBED_DROP, _EXPAND, _SUBBAND_KERNEL, AdaptiveFusion, AttnStatPool,
    Classifier, DropPath, Mamba, RMSNorm, SubbandStem, TFBlock,
)

# Symmetric stem kernel for the branches that no single subband owns
# (shared branch, raw front-end).
_SYM_KERNEL = (5, 5)


# ─── Sequence backbones (the #4/#5 variants) ──────────────────────────────────

class _MambaSeqLayer(nn.Module):
    """One Mamba layer: bidirectional (merge gate|add|concat) or unidirectional.

    With direction='bi', merge='gate' this is byte-for-byte `model.BiMambaLayer`
    (same submodules, same zero-init gate) — the anchor for the ours==WavMamba
    parameter check.
    """

    def __init__(self, d_model: int, d_state: int = 32, drop_path: float = 0.0,
                 direction: str = 'bi', merge: str = 'gate'):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                'mamba_ssm is required to build a Mamba backbone.\n'
                f'Original import error: {_MAMBA_IMPORT_ERROR}')
        self.direction = direction
        self.merge     = merge
        self.norm = RMSNorm(d_model)
        self.fwd  = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=_D_CONV, expand=_EXPAND)
        if direction == 'bi':
            self.bwd = Mamba(d_model=d_model, d_state=d_state,
                             d_conv=_D_CONV, expand=_EXPAND)
            if merge == 'gate':
                self.gate = nn.Linear(2 * d_model, d_model)
                nn.init.zeros_(self.gate.weight)   # g = 0.5 at init -> 0.5(f+b)
                nn.init.zeros_(self.gate.bias)
            elif merge == 'concat':
                self.proj = nn.Linear(2 * d_model, d_model)  # plain learned merge
            elif merge != 'add':
                raise ValueError(f"merge must be gate|add|concat, got {merge!r}")
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.fwd(h)
        if self.direction == 'uni':
            y = f
        else:
            b = self.bwd(h.flip(1)).flip(1)
            if self.merge == 'gate':
                g = torch.sigmoid(self.gate(torch.cat([f, b], dim=-1)))
                y = g * f + (1.0 - g) * b
            elif self.merge == 'add':
                y = 0.5 * (f + b)
            else:                                  # concat
                y = self.proj(torch.cat([f, b], dim=-1))
        return x + self.dp(y)


class _BiLSTMLayer(nn.Module):
    """Residual bidirectional-LSTM layer, drop-in for a Mamba layer.

    LSTM(d_model -> d_model//2, bidirectional) concatenates the two directions
    back to d_model, so no output projection is needed.
    """

    def __init__(self, d_model: int, drop_path: float = 0.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f'BiLSTM needs an even d_model; got {d_model}')
        self.norm = RMSNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True,
                            bidirectional=True)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()             # cuDNN contiguous-weights path
        y, _ = self.lstm(self.norm(x))             # (B, T, d_model)
        return x + self.dp(y)


class _Backbone(nn.Module):
    """Stack of sequence layers + final RMSNorm. Reproduces model.BiMamba for
    backbone='bimamba', merge='gate'."""

    def __init__(self, d_model: int, n_layers: int, d_state: int,
                 backbone: str, merge: str):
        super().__init__()
        if len(_DP_MAMBA) != n_layers:
            raise ValueError(f'len(_DP_MAMBA)={len(_DP_MAMBA)} != n_layers={n_layers}')
        if backbone == 'bilstm':
            layers = [_BiLSTMLayer(d_model, drop_path=_DP_MAMBA[i])
                      for i in range(n_layers)]
        else:
            direction = 'uni' if backbone == 'unimamba' else 'bi'
            layers = [_MambaSeqLayer(d_model, d_state=d_state,
                                     drop_path=_DP_MAMBA[i],
                                     direction=direction, merge=merge)
                      for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.norm   = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─── One branch: [stem + TFBlocks] -> flatten -> embed -> backbone ────────────

class _Branch(nn.Module):
    """(B, in_ch, T, F) -> (B, T, d_model).

    With stem: SubbandStem -> 3x TFBlock -> flatten -> embed (matches
    model.BranchBackbone + its external SubbandStem). Without stem (#3): flatten
    the raw maps straight into the embed, no convolution at all.
    """

    def __init__(self, in_ch: int, f2: int, d_model: int, d_stem: int,
                 d_state: int, n_mamba_layers: int, kernel: tuple,
                 use_stem: bool, backbone: str, merge: str):
        super().__init__()
        self.use_stem = use_stem
        if use_stem:
            self.stem   = SubbandStem(in_ch, d_stem, kernel=kernel)
            self.blocks = nn.ModuleList([
                TFBlock(d_stem, dilation=_DILATIONS[i], drop_path=_DP_CNN[i])
                for i in range(len(_DILATIONS))])
            embed_in = d_stem * f2
        else:
            embed_in = in_ch * f2
        self.embed = nn.Sequential(
            nn.Linear(embed_in, d_model), nn.SiLU(), nn.Dropout(_EMBED_DROP))
        self.backbone = _Backbone(d_model, n_mamba_layers, d_state,
                                  backbone=backbone, merge=merge)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_stem:
            x = self.stem(x)
            for blk in self.blocks:
                x = blk(x)
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)   # flatten F x C
        x = self.embed(x)                                 # (B, T, d_model)
        return self.backbone(x)


# ─── Full configurable model ──────────────────────────────────────────────────

class AblationWavMamba(nn.Module):
    """Configurable WavMamba for the ablation study. See module docstring.

    Args mirror WavMamba's dataset dims + width knobs, plus the seven ablation
    flags. Input: (B, C, T, F) where C = 2*n_antennas for front_end='dwt' (packed
    [HL|LH]) or C = n_antennas for front_end='raw'.
    """

    def __init__(self, num_classes: int = 7, n_antennas: int = 3, f2: int = 15,
                 d_model: int = 64, d_stem: int = 16, d_state: int = 32,
                 n_mamba_layers: int = 2,
                 front_end: str = 'dwt', branch: str = 'separate',
                 stem: str = 'stem', backbone: str = 'bimamba',
                 merge: str = 'gate', fusion: str = 'adaptive',
                 pool: str = 'attnstat'):
        super().__init__()
        for name, val, allowed in (
                ('front_end', front_end, ('dwt', 'raw')),
                ('branch',    branch,    ('separate', 'shared')),
                ('stem',      stem,      ('stem', 'nostem')),
                ('backbone',  backbone,  ('bimamba', 'unimamba', 'bilstm')),
                ('merge',     merge,     ('gate', 'add', 'concat')),
                ('fusion',    fusion,    ('adaptive', 'mean', 'concat')),
                ('pool',      pool,      ('attnstat', 'mean'))):
            if val not in allowed:
                raise ValueError(f'{name} must be one of {allowed}, got {val!r}')

        self.front_end   = front_end
        self.n_per_sub   = n_antennas
        self.f2          = f2
        use_stem         = (stem == 'stem')

        # How the input channels map to branches:
        #   raw            -> 1 branch over the whole (n_ant) map, symmetric kernel
        #   dwt + shared   -> 1 branch over the whole (2*n_ant) packed map, sym kernel
        #   dwt + separate -> 2 branches (HL, LH), each n_ant, physical kernels
        if front_end == 'raw':
            specs = [(n_antennas, _SYM_KERNEL)]
            self._separate = False
        elif branch == 'shared':
            specs = [(2 * n_antennas, _SYM_KERNEL)]
            self._separate = False
        else:
            specs = [(n_antennas, _SUBBAND_KERNEL['HL']),
                     (n_antennas, _SUBBAND_KERNEL['LH'])]
            self._separate = True

        self.branches = nn.ModuleList([
            _Branch(in_ch, f2, d_model, d_stem, d_state, n_mamba_layers,
                    kernel, use_stem, backbone, merge)
            for (in_ch, kernel) in specs])
        self.n_branches = len(specs)

        # Branch fusion — only when there is more than one branch.
        self._fusion = fusion if self.n_branches > 1 else None
        if self._fusion == 'adaptive':
            self.fusion = AdaptiveFusion(d_model, self.n_branches)
        elif self._fusion == 'concat':
            self.fusion = nn.Linear(self.n_branches * d_model, d_model)

        # Pooling over time.
        if pool == 'attnstat':
            self.tpool = AttnStatPool(d_model)
            head_in    = 2 * d_model
        else:
            self.tpool = None                      # mean-pool in forward
            head_in    = d_model
        self.head = Classifier(head_in, num_classes=num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 4:
            raise ValueError(f'expected 4-D input (B,C,T,F), got {tuple(X.shape)}')

        if self._separate:
            nps = self.n_per_sub
            streams = [self.branches[k](X[:, k * nps:(k + 1) * nps])
                       for k in range(self.n_branches)]
        else:
            streams = [self.branches[0](X)]

        if len(streams) == 1:
            z = streams[0]
        elif self._fusion == 'adaptive':
            z = self.fusion(streams)
        elif self._fusion == 'concat':
            z = self.fusion(torch.cat(streams, dim=-1))
        else:                                      # mean
            z = torch.stack(streams, dim=0).mean(dim=0)

        z = self.tpool(z) if self.tpool is not None else z.mean(dim=1)
        return self.head(z)


# ─── Registry ─────────────────────────────────────────────────────────────────
# One base config (OURS) + one single-flag override per row, exactly like the
# xrf55_bench LADDER. `front_end` lives inside kwargs and also selects the bench
# the driver builds. 11 rows: 7 axes, ours counted once.

OURS = dict(front_end='dwt', branch='separate', stem='stem', backbone='bimamba',
            merge='gate', fusion='adaptive', pool='attnstat')


def _v(**override) -> dict:
    """OURS with one flag overridden (rejects typos in the override key)."""
    bad = set(override) - set(OURS)
    if bad:
        raise KeyError(f'unknown ablation flag(s) {bad}; valid: {sorted(OURS)}')
    return {**OURS, **override}


ABLATIONS = {
    'ours':        dict(kwargs=_v(),                    note='full WavMamba (paper configuration)'),
    'a1_raw':      dict(kwargs=_v(front_end='raw'),     note='#1 no DWT — raw amplitude, single branch'),
    'a2_shared':   dict(kwargs=_v(branch='shared'),     note='#2 one shared branch (HL||LH), no per-subband specialisation'),
    'a3_nostem':   dict(kwargs=_v(stem='nostem'),       note='#3 no CNN stem/TFBlocks — DWT -> embed -> Mamba'),
    'a4_bilstm':   dict(kwargs=_v(backbone='bilstm'),   note='#4 BiLSTM backbone'),
    'a4_unimamba': dict(kwargs=_v(backbone='unimamba'), note='#4 unidirectional Mamba'),
    'a5_add':      dict(kwargs=_v(merge='add'),         note='#5 fwd/bwd merge = mean (f+b)/2'),
    'a5_concat':   dict(kwargs=_v(merge='concat'),      note='#5 fwd/bwd merge = linear on concat'),
    'a6_mean':     dict(kwargs=_v(fusion='mean'),       note='#6 branch fusion = mean'),
    'a6_concat':   dict(kwargs=_v(fusion='concat'),     note='#6 branch fusion = linear on concat'),
    'a7_mean':     dict(kwargs=_v(pool='mean'),         note='#7 mean-pool over time'),
}


def build_ablation_model(variant: str, meta: dict) -> AblationWavMamba:
    """Assemble the model for one registry variant, dims read from bench meta."""
    if variant not in ABLATIONS:
        raise KeyError(f'unknown variant {variant!r}; choose from {list(ABLATIONS)}')
    return AblationWavMamba(num_classes=meta['classes'], n_antennas=meta['n_ant'],
                            f2=meta['F2'], **ABLATIONS[variant]['kwargs'])


def variant_front_end(variant: str) -> str:
    """Which bench a variant consumes ('dwt' or 'raw')."""
    return ABLATIONS[variant]['kwargs']['front_end']


# ─── Aggregation ────────────────────────────────────────────────────────────
# Rediscover finished runs from disk (survives a kernel restart) and print the
# ablation table. Pure disk-glob over the deterministic output layout.

def ablation_table(dataset: str, out_root=None) -> str:
    """Markdown table of every finished ablation run for `dataset`.

    Globs <out_root>/outputs/ablation/<dataset>/*/metrics.json and builds one
    row per variant: acc, macro-F1, params_M, macs_M, note. Rows are ordered by
    the registry (ours first). Besides printing + returning the markdown, the
    aggregate is written next to the runs so the headline numbers survive as
    files, not just a log line:
        <base>/summary.md   the markdown table (paste-ready)
        <base>/summary.csv  the same rows for a spreadsheet / re-plotting
    """
    root = Path(out_root) if out_root else DATA_ROOT
    base = root / 'outputs' / 'ablation' / dataset
    found = {}
    for mpath in base.glob('*/metrics.json'):
        variant = mpath.parent.name
        with open(mpath) as f:
            m = json.load(f)
        found[variant] = m

    if not found:
        table = f'(no ablation runs found under {base})'
        print(table)
        return table

    order = [v for v in ABLATIONS if v in found] + \
            [v for v in found if v not in ABLATIONS]
    # Numeric records first; formatting for md/csv is derived from these.
    records = []
    for v in order:
        s = found[v].get('summary', {})
        records.append(dict(
            variant=v,
            acc=s.get('test_accuracy_mean'),
            f1=s.get('test_f1_macro_mean'),
            params_M=s.get('params_M'),
            macs_M=s.get('macs_M'),
            ssm_counted=s.get('macs_ssm_counted', True),
            note=ABLATIONS.get(v, {}).get('note', ''),
        ))

    def _pct(x): return f'{x * 100:.2f}' if x is not None else '—'
    def _f(x, p): return f'{x:.{p}f}' if x is not None else '—'

    header = ('| variant | acc | macro-F1 | params (M) | MACs (M) | note |\n'
              '|---------|-----|----------|-----------|----------|------|')
    rows = []
    for r in records:
        star = '' if r['ssm_counted'] else ' (no-ssm)'
        rows.append(f'| {r["variant"]} | {_pct(r["acc"])} | {_pct(r["f1"])} | '
                    f'{_f(r["params_M"], 3)} | {_f(r["macs_M"], 1)}{star} | {r["note"]} |')
    table = header + '\n' + '\n'.join(rows)

    base.mkdir(parents=True, exist_ok=True)
    with open(base / 'summary.md', 'w', encoding='utf-8') as f:
        f.write(f'# WavMamba ablation — {dataset}\n\n{table}\n')
    with open(base / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'acc', 'f1_macro', 'params_M', 'macs_M',
                    'ssm_counted', 'note'])
        for r in records:
            w.writerow([r['variant'], r['acc'], r['f1'], r['params_M'],
                        r['macs_M'], r['ssm_counted'], r['note']])

    print(table)
    return table
