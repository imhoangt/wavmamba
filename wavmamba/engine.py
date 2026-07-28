"""Training / evaluation primitives for WavMamba.

Stateless helpers used by trainer.py; each is callable on its own (e.g. to
evaluate a saved checkpoint without running the full seed loop).

    set_seed, configure_speed_mode   reproducibility + cuDNN/TF32 speed mode
    evaluate, evaluate_full          test-set metrics (quick / full + preds)
    count_macs, wavmamba_macs        analytic MAC counting (see that section)
    measure_efficiency               params, MACs, GPU latency
    make_optimizer/make_scheduler    build from a TrainCfg
    make_criterion                   the fixed paper loss (cross-entropy)
    train_epoch                      one optimization epoch
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim.lr_scheduler import LambdaLR


# ── Seeding + speed ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_speed_mode():
    """Faster, not bit-level deterministic: cuDNN auto-tuning + TF32 matmul."""
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    """Quick eval. Returns (acc, f1_macro)."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X.to(device)).argmax(1).cpu().tolist()
            gts   += y.tolist()
    return accuracy_score(gts, preds), f1_score(gts, preds, average='macro')


def evaluate_full(model, loader, device, num_classes):
    """Full eval. Returns (acc, f1, f1_per_cls, cm, preds, probs, gts)."""
    model.eval()
    preds, probs, gts = [], [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            probs  += torch.softmax(logits, 1).cpu().numpy().tolist()
            preds  += logits.argmax(1).cpu().tolist()
            gts    += y.tolist()
    acc        = accuracy_score(gts, preds)
    f1         = f1_score(gts, preds, average='macro')
    f1_per_cls = f1_score(gts, preds, average=None, labels=list(range(num_classes))).tolist()
    cm         = confusion_matrix(gts, preds, labels=list(range(num_classes))).tolist()
    return acc, f1, f1_per_cls, cm, preds, probs, gts


# ── MAC counting ──────────────────────────────────────────────────────────────
# Why an explicit counter instead of `fvcore`:
# `fvcore.nn.FlopCountAnalysis` counts by matching ATen operators. Mamba's fast
# path (`mamba_inner_fn`) performs the selective scan in a compiled CUDA
# extension (`selective_scan_cuda`), and it applies `in_proj` / `x_proj` /
# `dt_proj` / `out_proj` as raw `F.linear` calls on weight tensors *inside* a
# custom `autograd.Function` rather than as `nn.Linear` module calls. Neither
# the scan nor those projections is a plain ATen op reachable from the module
# graph, so an operator-matching counter silently reports 0 for them — i.e. it
# drops the entire state-space core, the dominant cost of this model (~72% of
# WavMamba's MACs at the default UT-HAR dimensions).
#
# Approach — hybrid, so every number is either measured or read off the live
# model:
# * Everything OUTSIDE the Mamba blocks (stem convs, axial depthwise convs,
#   pointwise convs, embed / gate / fusion / pooling / head linears) is counted
#   from REAL tensor shapes captured by forward hooks. Nothing is assumed.
# * Everything INSIDE each Mamba block is counted in closed form from that
#   block's own attributes (`d_model`, `d_inner`, `d_state`, `d_conv`,
#   `dt_rank`), so the count cannot drift from the configuration actually run.
#
# Convention (stated explicitly because the literature is not consistent):
# * 1 MAC = one multiply-accumulate. FLOPs are reported as 2 x MACs.
# * Only multiply-accumulate work is counted: matmuls, convolutions, and the
#   selective-scan recurrence. Normalisations, activations, dropout, flips,
#   reshapes and elementwise gate products are NOT counted — all O(L*D) while
#   the counted terms are O(L*D*N) or matmul-sized.
# * Selective scan: 4 MACs per (d_inner, L, d_state) element, matching the
#   published Vision Mamba complexity Omega(SSM) = 3*M*(2D)*N + M*(2D)*N
#   (arXiv:2401.09417, Sec. 3.5), with M = sequence length, 2D = d_inner,
#   N = d_state. The reference implementation `selective_scan_ref` contains one
#   further elementwise multiply in `delta * B * u`, so a 5-per-element
#   convention is also defensible (+25% on the scan term); the 4-per-element
#   form is used because it is the citable one. The active value is
#   SCAN_MACS_PER_ELEMENT.
# * Batch size 1. Inference only (no backward).

# See the convention note above: Vision Mamba's Omega(SSM) = 3*M*(2D)*N + M*(2D)*N.
SCAN_MACS_PER_ELEMENT = 4

CONVENTION = (
    '1 MAC = 1 multiply-accumulate; FLOPs = 2 x MACs; batch=1, inference. '
    'Counts matmuls, convolutions and the selective-scan recurrence '
    f'({SCAN_MACS_PER_ELEMENT} MACs per d_inner*L*d_state element, per Vision '
    'Mamba arXiv:2401.09417 Sec. 3.5). Excludes norms, activations, dropout '
    'and elementwise gate products (all O(L*D)). Mamba-internal projections '
    'and the scan are counted in closed form because the fast path hides them '
    'from operator-level counters; everything else from measured shapes.'
)


def _is_mamba(module: nn.Module) -> bool:
    """A real mamba_ssm.Mamba block, identified by the attributes we need.

    Duck-typed rather than isinstance-checked so the counter also works when
    mamba_ssm is unavailable (e.g. the CPU test double), in which case the
    block's own layers are counted by the generic hooks instead.
    """
    return all(hasattr(module, a) for a in
               ('d_model', 'd_inner', 'd_state', 'd_conv', 'dt_rank'))


def _mamba_macs(m: nn.Module, seq_len: int) -> dict:
    """Closed-form MACs for one Mamba block over `seq_len` steps, batch 1."""
    L  = seq_len
    dm = int(m.d_model)
    D  = int(m.d_inner)
    N  = int(m.d_state)
    K  = int(m.d_conv)
    R  = int(m.dt_rank)
    return {
        # in_proj: d_model -> 2*d_inner (x and z), at every step.
        'in_proj':   L * dm * 2 * D,
        # depthwise causal conv1d over d_inner channels, kernel d_conv.
        'conv1d':    L * D * K,
        # x_proj: d_inner -> dt_rank + 2*d_state (delta, B, C).
        'x_proj':    L * D * (R + 2 * N),
        # dt_proj: dt_rank -> d_inner.
        'dt_proj':   L * R * D,
        # selective scan recurrence — see SCAN_MACS_PER_ELEMENT.
        'scan':      SCAN_MACS_PER_ELEMENT * L * D * N,
        # NOT counted: the u*D skip and the SiLU(z) output gate — elementwise
        # O(L*D) products, excluded by the stated convention.
        # out_proj: d_inner -> d_model.
        'out_proj':  L * D * dm,
    }


def _lstm_macs(m: nn.LSTM, seq_len: int) -> int:
    """Closed-form MACs for one nn.LSTM over `seq_len` steps, batch 1.

    nn.LSTM is a FUSED module — its gate weights (weight_ih/weight_hh) are not
    nn.Linear children, so the generic hooks never see them and the whole
    recurrent core would be silently missed (the same trap that hides Mamba's
    fast path). Counted here in closed form instead.

    Per direction, per step: the four gates apply W_ih (in_size -> hidden) and
    W_hh (hidden -> hidden), i.e. 4 * (in_size + hidden) * hidden MACs. The
    internal gate elementwise products (O(hidden)) are excluded, matching the
    convention used for Mamba's gate. Stacked / bidirectional layers are summed;
    layer>0 takes hidden*num_directions as its input size.
    """
    H     = int(m.hidden_size)
    dirs  = 2 if m.bidirectional else 1
    total = 0
    for layer in range(int(m.num_layers)):
        in_size = int(m.input_size) if layer == 0 else H * dirs
        total += seq_len * dirs * 4 * (in_size + H) * H
    return total


def count_macs(model: nn.Module, inputs) -> tuple[int, dict]:
    """Count inference MACs for one forward pass.

    Args:
        model  : the model to measure (left in its original training mode).
        inputs : tuple of input tensors, batch size 1.

    Returns:
        (total_macs, breakdown) where breakdown maps a component label to its
        MAC count.
    """
    mamba_blocks = {id(m): m for m in model.modules() if _is_mamba(m)}

    # Layers owned by a Mamba block are accounted for in closed form; make sure
    # the generic hooks never double-count them (they DO fire when mamba_ssm
    # runs its non-fast path).
    inner: set[int] = set()
    for blk in mamba_blocks.values():
        inner.update(id(sub) for sub in blk.modules() if sub is not blk)

    tally: dict[str, int] = {}
    handles = []

    def _add(label: str, n: int):
        tally[label] = tally.get(label, 0) + int(n)

    def _hook_linear(mod, args, out):
        # One MAC per (output element, input feature); bias adds are not MACs.
        n_pos = out.numel() // out.shape[-1]
        _add('linear', n_pos * mod.in_features * mod.out_features)

    def _hook_conv(mod, args, out):
        # Per output element: (in_channels/groups) * prod(kernel).
        k = 1
        for d in mod.kernel_size:
            k *= d
        _add('conv', out.numel() * (mod.in_channels // mod.groups) * k)

    for mod in model.modules():
        if id(mod) in inner or id(mod) in mamba_blocks:
            continue
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(_hook_linear))
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(mod.register_forward_hook(_hook_conv))

    # Sequence length reaching each Mamba block / LSTM, captured not assumed.
    seq_lens: dict[int, int] = {}

    def _hook_seq(mod, args, out):
        x = args[0]
        seq_lens[id(mod)] = int(x.shape[1])       # (B, L, d)

    for mid, blk in mamba_blocks.items():
        handles.append(blk.register_forward_hook(_hook_seq))

    # nn.LSTM: fused, no Linear children — count in closed form like Mamba.
    lstm_blocks = {id(m): m for m in model.modules() if isinstance(m, nn.LSTM)}
    for lid, lstm in lstm_blocks.items():
        handles.append(lstm.register_forward_hook(_hook_seq))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    for mid, blk in mamba_blocks.items():
        if mid not in seq_lens:
            continue                              # block not reached this pass
        for label, n in _mamba_macs(blk, seq_lens[mid]).items():
            _add(f'mamba.{label}', n)

    for lid, lstm in lstm_blocks.items():
        if lid not in seq_lens:
            continue
        _add('lstm', _lstm_macs(lstm, seq_lens[lid]))

    total = sum(tally.values())
    return total, dict(sorted(tally.items()))


def wavmamba_macs(model: nn.Module, input_shape) -> dict:
    """MACs for one batch-1 forward pass, as a JSON-serialisable report.

    Args:
        model       : the model to measure (mode and device left untouched).
        input_shape : input shape WITHOUT the batch dim, e.g. (C, T2, F2).

    Returns:
        {'total_M', 'flops_M', 'breakdown_M', 'ssm_counted', 'note'} with counts
        in millions. 'note' carries the counting convention verbatim so a
        reported number is never separated from its definition.

        'ssm_counted' reports whether the recurrent SEQUENCE CORE was counted.
        It is True for a real Mamba block (closed-form scan + projections) and
        for an nn.LSTM backbone (closed-form gates — the a4_bilstm ablation,
        which runs natively on CPU). It is False only when neither is found —
        i.e. a Mamba model with the CPU test double swapped in, whose total is
        then missing the state-space core. Do not report an ssm_counted=False run.
    """
    device = next(model.parameters()).device
    x = torch.randn(1, *tuple(input_shape), device=device)
    total, tally = count_macs(model, (x,))
    # The sequence core is counted iff a Mamba block (mamba.*) or an LSTM (lstm)
    # term is present. Neither => a Mamba model ran under the CPU test double.
    ssm_counted = any(k.startswith('mamba.') for k in tally) or ('lstm' in tally)
    note = CONVENTION if ssm_counted else (
        'INCOMPLETE — no mamba_ssm.Mamba block was found (CPU test double in '
        'use), so the state-space core is NOT included. Do not report. '
        + CONVENTION
    )
    return {
        'total_M':     round(total / 1e6, 3),
        'flops_M':     round(2 * total / 1e6, 3),
        'breakdown_M': {k: round(v / 1e6, 3) for k, v in tally.items()},
        'ssm_counted': ssm_counted,
        'note':        note,
    }


# ── Efficiency probe ──────────────────────────────────────────────────────────

def measure_efficiency(model, device, input_shape):
    """Params (M), MACs (M), GPU inference latency (ms), all at batch size 1.

    MACs come from wavmamba_macs() above — an explicit analytic count, NOT a
    tracer. See the MAC-counting section for why tracers undercount this model.

    Args:
        input_shape: model input shape WITHOUT the batch dim: (C, T2, F2).
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    macs = wavmamba_macs(model, input_shape)

    if device.type == 'cuda':
        x = torch.randn(1, *input_shape, device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(50):
                model(x)
        timings = []
        with torch.no_grad():
            for _ in range(200):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); model(x); e.record()
                torch.cuda.synchronize()
                timings.append(s.elapsed_time(e))
        lat_mean = round(float(np.mean(timings)), 2)
        lat_std  = round(float(np.std(timings)), 2)
    else:
        lat_mean = lat_std = None

    return params_m, macs, lat_mean, lat_std


# ── Weight-decay exclusion ────────────────────────────────────────────────────

_NO_DECAY_KEYS  = {'bias', 'A_log', 'D'}
_NORM_MODULES   = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                   nn.BatchNorm3d, nn.GroupNorm)


def _build_no_decay_set(model: nn.Module) -> set:
    """Param names that must NOT receive weight decay: norm-layer params,
    Mamba SSM A_log/D, and all biases. Matched by leaf name."""
    no_decay: set = set()
    for mn, m in model.named_modules():
        is_norm = isinstance(m, _NORM_MODULES) or type(m).__name__ == 'RMSNorm'
        if is_norm:
            for pn, _ in m.named_parameters(recurse=False):
                no_decay.add(f'{mn}.{pn}' if mn else pn)
    for pn, _ in model.named_parameters():
        if pn.split('.')[-1] in _NO_DECAY_KEYS:
            no_decay.add(pn)
    return no_decay


# ── Optimizer / scheduler / criterion ─────────────────────────────────────────

def make_optimizer(model: nn.Module, cfg):
    """Build the optimizer from a TrainCfg (see config.py)."""
    if cfg.wd_exclude_norm_bias:
        no_decay   = _build_no_decay_set(model)
        decay_p    = [p for n, p in model.named_parameters()
                      if p.requires_grad and n not in no_decay]
        no_decay_p = [p for n, p in model.named_parameters()
                      if p.requires_grad and n in no_decay]
        params = [
            {'params': decay_p,    'weight_decay': cfg.weight_decay},
            {'params': no_decay_p, 'weight_decay': 0.0},
        ]
        wd_kw = {}
    else:
        params = model.parameters()
        wd_kw  = {'weight_decay': cfg.weight_decay}

    opt_kw = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, **wd_kw)
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(params, **opt_kw)
    if cfg.optimizer == 'adam':
        return torch.optim.Adam(params, **opt_kw)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def make_scheduler(optimizer, cfg):
    """Build the LR scheduler from a TrainCfg (see config.py)."""
    if cfg.scheduler is None:
        return None
    if cfg.scheduler == 'warmup_cosine':
        W           = cfg.warmup_epochs
        T           = cfg.num_epochs
        floor_ratio = cfg.floor_lr / cfg.lr

        def _lr_lambda(epoch):
            if epoch < W:
                return (epoch + 1) / max(W, 1)          # linear warmup
            progress = min((epoch - W + 1) / max(T - W, 1), 1.0)
            cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cos_val

        return LambdaLR(optimizer, _lr_lambda)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")


def make_criterion():
    """Fixed paper loss: plain cross-entropy (no label smoothing)."""
    return nn.CrossEntropyLoss()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    """Run one epoch. Returns (avg_loss, avg_grad_norm). grad_clip=None: no clip."""
    model.train()
    total_loss = 0.0
    grad_norms = []
    max_norm   = grad_clip if grad_clip is not None else float('inf')
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        grad_norms.append(
            nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(loader), float(np.mean(grad_norms))
