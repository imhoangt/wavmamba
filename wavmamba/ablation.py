"""Ablation study for WavMamba — a parallel, configurable assembler.

The shipped `model.WavMamba` is deliberately locked (it raises on any deviation
from the paper flags), so it cannot host the ablation variants. This module
provides `AblationWavMamba`, a parallel assembler that takes the SAME leaf
blocks from `model.py` (`SubbandStem`, `TFBlock`, `RMSNorm`, `AdaptiveFusion`,
`AttnStatPool`, `Classifier`) and composes them under real, swappable flags.
The paper model and its reproducibility guarantee are untouched.

Seven single-variable axes, each swapping exactly one component vs a centre
configuration. See `ABLATIONS` / `ABLATIONS_S` for the two registries.

    front_end : 'dwt'  (Haar HL+LH, ours)         | 'raw' (no DWT, single map)
    branch    : 'separate' (per-subband, ours)    | 'shared' (one branch)
                | 'split' (one branch, per-subband stems)
    stem      : 'stem' (SubbandStem+3xTFBlock, ours) | 'nostem' (embed straight)
    backbone  : 'bimamba' (ours) | 'unimamba' | 'bilstm'
    merge     : 'gate' (per-channel zero-init, ours) | 'add' | 'concat'
    fusion    : 'adaptive' (softmax gate, ours) | 'mean' | 'concat'
    pool      : 'attnstat' (ECAPA, ours) | 'statpool' (no attention) | 'mean'

Two registries share this assembler:

    ABLATIONS    centre = 'ours'   (dwt, separate)  — the paper model
    ABLATIONS_S  centre = 'center' (dwt, shared)    — WavMamba-S, the efficient
                 variant. Three of its rows are byte-identical configurations of
                 ABLATIONS rows and reuse those runs; a module-level check
                 (`_REUSED`) fails the import if that stops being true.

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


# ─── Stem / pooling variants that are not just a flag on an existing block ────

class _SplitStem(nn.Module):
    """Per-subband stems with physical kernels, concatenated to d_stem channels.

    (B, 2*n_per_sub, T, F) packed [HL|LH] -> (B, d_stem, T, F). Each subband gets
    its own SubbandStem(n_per_sub -> d_stem//2) with its own physically-motivated
    kernel; the halves are concatenated so the downstream TFBlock stack and embed
    are identical to the shared branch. Isolates "specialised kernels" from
    "duplicated backbone".

    Note this is a 3-change vs the shared stem, not a single variable: directional
    kernels, each stem seeing only its own subband, AND half the output width per
    stem (forced by keeping d_stem fixed so everything downstream is unchanged).
    """

    def __init__(self, n_per_sub: int, d_stem: int = 16):
        super().__init__()
        if d_stem % 2 != 0:
            raise ValueError(f'_SplitStem needs an even d_stem; got {d_stem}')
        self.n_per_sub = n_per_sub
        self.stems = nn.ModuleList([
            SubbandStem(n_per_sub, d_stem // 2, kernel=_SUBBAND_KERNEL[s])
            for s in ('HL', 'LH')])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nps = self.n_per_sub
        return torch.cat([stem(x[:, k * nps:(k + 1) * nps])
                          for k, stem in enumerate(self.stems)], dim=1)


class _StatPool(nn.Module):
    """Unweighted [mean || std] over time — AttnStatPool with attention removed.

    Output 2*d, so the classifier is unchanged. AttnStatPool's score net is
    zero-init => uniform weights at step 0, so this module is EXACTLY
    AttnStatPool at initialisation — the same "frozen at init" relationship
    merge='add' has with the merge gate. That is what makes it a clean ablation
    of learned temporal weighting alone, with the std branch held fixed
    (pool='mean' drops both at once).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)                                   # (B, d)
        var  = x.var(dim=1, unbiased=False)                    # (B, d)
        return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)


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
    the raw maps straight into the embed, no convolution at all. With
    split_stem: `_SplitStem` replaces the single stem, same output width.
    """

    def __init__(self, in_ch: int, f2: int, d_model: int, d_stem: int,
                 d_state: int, n_mamba_layers: int, kernel: tuple,
                 use_stem: bool, backbone: str, merge: str,
                 split_stem: bool = False):
        super().__init__()
        self.use_stem = use_stem
        if use_stem:
            self.stem   = (_SplitStem(in_ch // 2, d_stem) if split_stem else
                           SubbandStem(in_ch, d_stem, kernel=kernel))
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
                ('branch',    branch,    ('separate', 'shared', 'split')),
                ('stem',      stem,      ('stem', 'nostem')),
                ('backbone',  backbone,  ('bimamba', 'unimamba', 'bilstm')),
                ('merge',     merge,     ('gate', 'add', 'concat')),
                ('fusion',    fusion,    ('adaptive', 'mean', 'concat')),
                ('pool',      pool,      ('attnstat', 'statpool', 'mean'))):
            if val not in allowed:
                raise ValueError(f'{name} must be one of {allowed}, got {val!r}')

        self.front_end   = front_end
        self.n_per_sub   = n_antennas
        self.f2          = f2
        use_stem         = (stem == 'stem')

        # How the input channels map to branches:
        #   raw            -> 1 branch over the whole (n_ant) map, symmetric kernel
        #   dwt + shared   -> 1 branch over the whole (2*n_ant) packed map, sym kernel
        #   dwt + split    -> 1 branch as above, but per-subband stems inside it
        #   dwt + separate -> 2 branches (HL, LH), each n_ant, physical kernels
        split_stem = False
        if front_end == 'raw':
            specs = [(n_antennas, _SYM_KERNEL)]
            self._separate = False
        elif branch in ('shared', 'split'):
            specs = [(2 * n_antennas, _SYM_KERNEL)]
            self._separate = False
            split_stem = (branch == 'split')
        else:
            specs = [(n_antennas, _SUBBAND_KERNEL['HL']),
                     (n_antennas, _SUBBAND_KERNEL['LH'])]
            self._separate = True

        self.branches = nn.ModuleList([
            _Branch(in_ch, f2, d_model, d_stem, d_state, n_mamba_layers,
                    kernel, use_stem, backbone, merge, split_stem=split_stem)
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
        elif pool == 'statpool':
            self.tpool = _StatPool()               # same 2*d output, no params
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


# ─── Registry S: centre = WavMamba-S (one shared branch) ──────────────────────
# Same seven axes, re-centred on the efficient single-branch variant. Two things
# this buys over ABLATIONS: the DWT comparison becomes single-variable (with a
# one-branch centre, 'raw' differs only by the DWT — see the specs block above),
# and 'split'/'statpool' become expressible as one-flag moves.

CENTER = dict(front_end='dwt', branch='shared', stem='stem', backbone='bimamba',
              merge='gate', fusion='adaptive', pool='attnstat')


def _vs(**override) -> dict:
    """CENTER with one flag overridden (rejects typos in the override key)."""
    bad = set(override) - set(CENTER)
    if bad:
        raise KeyError(f'unknown ablation flag(s) {bad}; valid: {sorted(CENTER)}')
    return {**CENTER, **override}


ABLATIONS_S = {
    'center':      dict(kwargs=_vs(),                    note='WavMamba-S (one shared branch)'),
    'c1_raw':      dict(kwargs=_vs(front_end='raw'),     note='#1 no DWT — raw amplitude'),
    'c2_separate': dict(kwargs=_vs(branch='separate'),   note='#2 two per-subband branches + late fusion (= full WavMamba)'),
    'c3_split':    dict(kwargs=_vs(branch='split'),      note='#3 per-subband stems, one shared backbone'),
    'c4_nostem':   dict(kwargs=_vs(stem='nostem'),       note='#4 no CNN stem/TFBlocks — DWT -> embed -> Mamba'),
    'c5_uni':      dict(kwargs=_vs(backbone='unimamba'), note='#5 unidirectional Mamba'),
    'c5_bilstm':   dict(kwargs=_vs(backbone='bilstm'),   note='#5 BiLSTM backbone'),
    'c6_add':      dict(kwargs=_vs(merge='add'),         note='#6 fwd/bwd merge = mean (f+b)/2'),
    'c6_concat':   dict(kwargs=_vs(merge='concat'),      note='#6 fwd/bwd merge = linear on concat'),
    'c7_statpool': dict(kwargs=_vs(pool='statpool'),     note='#7 unweighted [mean||std] — attention removed'),
    'c7_mean':     dict(kwargs=_vs(pool='mean'),         note='#7 mean-pool over time'),
}


# Three ABLATIONS_S rows are configurations that ABLATIONS already ran, so their
# runs are reused (copied on disk) instead of re-trained. That is only sound
# while the configs keep building the same model — this check makes a drifting
# CENTER an ImportError instead of a mislabelled column.
_REUSED = (
    # (new, old, keys that do not affect the model that gets built)
    ('center',      'a2_shared', ()),
    ('c2_separate', 'ours',      ()),
    # front_end='raw' forces one branch and IGNORES `branch` (see the specs block
    # in AblationWavMamba.__init__), so these two build the same model despite
    # differing on that key.
    ('c1_raw',      'a1_raw',    ('branch',)),
)
for _new, _old, _skip in _REUSED:
    _a = {k: v for k, v in ABLATIONS_S[_new]['kwargs'].items() if k not in _skip}
    _b = {k: v for k, v in ABLATIONS[_old]['kwargs'].items() if k not in _skip}
    if _a != _b:
        raise AssertionError(
            f'{_new} must build the same model as {_old} — reused runs would be '
            f'mislabelled. Differences: '
            f'{ {k: (_a.get(k), _b.get(k)) for k in _a.keys() | _b.keys() if _a.get(k) != _b.get(k)} }')
del _new, _old, _skip, _a, _b


def build_ablation_model(variant: str, meta: dict,
                         registry=None) -> AblationWavMamba:
    """Assemble the model for one registry variant, dims read from bench meta.

    `registry` defaults to ABLATIONS; pass ABLATIONS_S for the WavMamba-S study.
    """
    reg = ABLATIONS if registry is None else registry
    if variant not in reg:
        raise KeyError(f'unknown variant {variant!r}; choose from {list(reg)}')
    return AblationWavMamba(num_classes=meta['classes'], n_antennas=meta['n_ant'],
                            f2=meta['F2'], **reg[variant]['kwargs'])


def variant_front_end(variant: str, registry=None) -> str:
    """Which bench a variant consumes ('dwt' or 'raw')."""
    reg = ABLATIONS if registry is None else registry
    return reg[variant]['kwargs']['front_end']


# ─── Aggregation ────────────────────────────────────────────────────────────
# Rediscover finished runs from disk (survives a kernel restart) and print the
# ablation table. Pure disk-glob over the deterministic output layout.

def ablation_table(dataset: str, base=None, registry=None) -> str:
    """Markdown table of every finished ablation run for `dataset`.

    Globs <base>/*/metrics.json and builds one row per variant: acc +- std,
    macro-F1 +- std, params_M, macs_M, latency, note. Rows are ordered by the
    registry (its centre first). Besides printing + returning the markdown, the
    aggregate is written next to the runs so the headline numbers survive as
    files, not just a log line:
        <base>/summary.md   the markdown table (paste-ready)
        <base>/summary.csv  the same rows for a spreadsheet / re-plotting

    `base` is the directory holding one subdirectory per variant. It defaults to
    the study-A location, DATA_ROOT/outputs/ablation/<dataset>; the driver passes
    its own out_dir parent, and the local archive (output/<DATASET>/<subdir>/
    <dataset>/) can be pointed at directly. One path argument, so a table can
    never be built from one layout and written into another.
    """
    reg  = ABLATIONS if registry is None else registry
    base = Path(base) if base is not None else \
        DATA_ROOT / 'outputs' / 'ablation' / dataset
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

    order = [v for v in reg if v in found] + \
            [v for v in found if v not in reg]
    # Numeric records first; formatting for md/csv is derived from these.
    records = []
    for v in order:
        s = found[v].get('summary', {})
        records.append(dict(
            variant=v,
            acc=s.get('test_accuracy_mean'),
            acc_std=s.get('test_accuracy_std'),
            f1=s.get('test_f1_macro_mean'),
            f1_std=s.get('test_f1_macro_std'),
            params_M=s.get('params_M'),
            macs_M=s.get('macs_M'),
            lat=s.get('latency_mean_ms'),
            lat_std=s.get('latency_std_ms'),
            ssm_counted=s.get('macs_ssm_counted', True),
            note=reg.get(v, {}).get('note', ''),
        ))

    def _pct(x): return f'{x * 100:.2f}' if x is not None else '—'
    def _f(x, p): return f'{x:.{p}f}' if x is not None else '—'

    def _pm(x, sd):
        """'99.12 +- 0.31' in percent; the std columns come straight from summary."""
        if x is None:
            return '—'
        return _pct(x) if sd is None else f'{_pct(x)} ± {_pct(sd)}'

    def _lat(x, sd):
        if x is None:
            return '—'
        return _f(x, 2) if sd is None else f'{_f(x, 2)} ± {_f(sd, 2)}'

    header = ('| variant | acc | macro-F1 | params (M) | MACs (M) | latency (ms) | note |\n'
              '|---------|-----|----------|-----------|----------|--------------|------|')
    rows = []
    for r in records:
        star = '' if r['ssm_counted'] else ' (no-ssm)'
        rows.append(f'| {r["variant"]} | {_pm(r["acc"], r["acc_std"])} | '
                    f'{_pm(r["f1"], r["f1_std"])} | '
                    f'{_f(r["params_M"], 3)} | {_f(r["macs_M"], 1)}{star} | '
                    f'{_lat(r["lat"], r["lat_std"])} | {r["note"]} |')
    table = header + '\n' + '\n'.join(rows)

    # The centre row is the registry's first key, and its kwargs name the study
    # unambiguously — two summary.md files for the same dataset would otherwise
    # be indistinguishable once they leave this directory.
    centre = next(iter(reg))
    cfg    = reg[centre]['kwargs']
    ident  = '/'.join(str(cfg[k]) for k in ('front_end', 'branch', 'stem',
                                            'backbone', 'merge', 'fusion', 'pool'))

    base.mkdir(parents=True, exist_ok=True)
    with open(base / 'summary.md', 'w', encoding='utf-8') as f:
        f.write(f'# WavMamba ablation — {dataset}\n\n'
                f'Centre: `{centre}` ({ident}). Every other row overrides exactly '
                f'one of those flags.\n\n{table}\n')
    with open(base / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'acc', 'acc_std', 'f1_macro', 'f1_macro_std',
                    'params_M', 'macs_M', 'latency_ms', 'latency_std_ms',
                    'ssm_counted', 'note'])
        for r in records:
            w.writerow([r['variant'], r['acc'], r['acc_std'], r['f1'],
                        r['f1_std'], r['params_M'], r['macs_M'], r['lat'],
                        r['lat_std'], r['ssm_counted'], r['note']])

    print(table)
    return table
