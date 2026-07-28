"""
WavMamba — WiFi-CSI human activity recognition
==============================================

A multi-branch (one branch per DWT subband) CNN + bidirectional-Mamba model
with adaptive late fusion, for WiFi-CSI human activity recognition on UT-HAR
and NTU-Fi.

This file ships the single architecture used in the paper:
    subbands  = ('HL', 'LH')   Haar 2-branch (no LL)
    pool      = 'attnstat'     attentive statistics pooling (ECAPA-style)
    stem_norm = False          no GroupNorm in the stem
    fusion    = 'gate'         per-channel gate fusion

Only the dataset-dependent dimensions (num_classes, n_antennas, f2) and four
width knobs (d_model, d_stem, d_state, n_mamba_layers) are configurable; the
architecture flags above are fixed and any attempt to change them raises
ValueError.

Design lineage: TF-Mamba (dual-stream DWT Mamba), Vim / HARMamba
(bidirectional SSM), PTM-Mamba (bidirectional gated Mamba), PerceptionNet
(late fusion > early fusion in HAR), ECAPA-TDNN (attentive stats pooling).

Input  : X (B, 2*n_antennas, T2, F2)  — packed Haar [HL | LH], subband-major.
         (T_raw -> T2 = T_raw//2, F_raw -> F2 = F_raw//2 after 2-D Haar DWT.)

Per-branch (one per selected subband s in {HL, LH}):
    subband_s (B, n_per_sub, T2, F2)
      -> Stem_s         per-subband conv (HL:(3,7), LH:(7,3)) + SiLU
      -> TFBlock x3     dilation [1,2,4] -> RF 7,19,43 timesteps
      -> flatten F x C  (B, d_stem, T2, F2) -> (B, T2, d_stem*F2)   (keep freq)
      -> Linear embed   d_stem*F2 -> d_model + SiLU + Dropout(0.1)
      -> BiMamba x2     fwd/bwd merged by a per-channel ZERO-INIT gate
    => S_s (B, T2, d_model)

Fusion + head:
    AdaptiveFusion(S_HL, S_LH)  per-channel gate (zero-init -> mean at step 0)
      -> AttnStatPool          ECAPA mean||std over time -> (B, 2*d_model)
      -> Classifier            LN -> Dropout -> Linear -> logits
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    _MAMBA_IMPORT_ERROR = None
except Exception as e:                                   # pragma: no cover
    Mamba = None
    HAS_MAMBA = False
    _MAMBA_IMPORT_ERROR = e


# Physically-motivated stem kernel per subband (the two detail subbands used).
_SUBBAND_KERNEL = {'HL': (3, 7), 'LH': (7, 3)}
#   HL (3,7) temporal burst onsets · LH (7,3) Doppler

# Locked architecture — the four values the paper commits to.
_LOCKED_SUBBANDS  = ('HL', 'LH')
_LOCKED_POOL      = 'attnstat'
_LOCKED_STEM_NORM = False
_LOCKED_FUSION    = 'gate'

# Fixed hyperparameters (not exposed — the paper uses these values).
_DILATIONS   = (1, 2, 4)            # TFBlock dilations -> RF 7, 19, 43 timesteps
_DP_CNN      = (0.0, 0.05, 0.1)     # DropPath per TFBlock
_DP_MAMBA    = (0.0, 0.10)          # DropPath per BiMamba layer
_D_CONV      = 4                    # Mamba local conv1d width
_EXPAND      = 2                    # Mamba inner-expansion factor
_EMBED_DROP  = 0.1                  # dropout after Linear embed
_CLF_DROPOUT = 0.2                  # classifier dropout


def _gn_groups(d: int) -> int:
    """GroupNorm groups targeting ~8 channels per group, guaranteed to divide d."""
    for g in range(max(1, d // 8), 0, -1):
        if d % g == 0:
            return g
    return 1


# ─── Stochastic depth ─────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop whole samples from a residual branch during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


# ─── CNN: per-subband stem + light axial-depthwise block ──────────────────────

class SubbandStem(nn.Module):
    """Conv2d(in_ch -> d_stem) + SiLU, one per subband. (stem_norm=False — no GN.)"""

    def __init__(self, in_ch: int, d_stem: int = 16, kernel=(5, 5)):
        super().__init__()
        kt, kf = kernel
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_stem, (kt, kf),
                      stride=(1, 1),
                      padding=(kt // 2, kf // 2)),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TFBlock(nn.Module):
    """Pre-norm sequential axial-depthwise block for 2-D (time x freq) maps.

        y   = GroupNorm(x)
        y   = dw_f(dw_t(y))        # sequential axial: temporal first, then freq
        y   = pw(SiLU(y))          # pointwise channel projection
        out = x + DropPath(y)
    """

    def __init__(self, d: int, k_t: int = 7, k_f: int = 3,
                 dilation: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(_gn_groups(d), d)
        self.dw_t = nn.Conv2d(d, d, (k_t, 1),
                              padding=(k_t // 2 * dilation, 0),
                              dilation=(dilation, 1), groups=d)
        self.dw_f = nn.Conv2d(d, d, (1, k_f), padding=(0, k_f // 2), groups=d)
        self.act  = nn.SiLU()
        self.pw   = nn.Conv2d(d, d, 1)
        self.dp   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw_f(self.dw_t(y))
        y = self.pw(self.act(y))
        return x + self.dp(y)


# ─── Bidirectional Mamba with per-channel zero-init gated fwd/bwd merge ────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


class BiMambaLayer(nn.Module):
    """One bidirectional Mamba layer with a per-channel zero-init gate.

        h = RMSNorm(x)
        f = Mamba_fwd(h)
        b = flip(Mamba_bwd(flip(h)))
        g = sigmoid(W·[f || b] + c)   # W=0, c=0 at init => g = 0.5
        y = g * f + (1 - g) * b        # starts as 0.5(f+b), specialises later
        x = x + DropPath(y)
    """

    def __init__(self, d_model: int, d_state: int = 32, drop_path: float = 0.0):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required to build BiMambaLayer.\n"
                "Install: pip install mamba-ssm[causal-conv1d] --no-build-isolation\n"
                f"Original import error: {_MAMBA_IMPORT_ERROR}"
            )
        self.norm = RMSNorm(d_model)
        self.fwd  = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=_D_CONV, expand=_EXPAND)
        self.bwd  = Mamba(d_model=d_model, d_state=d_state,
                          d_conv=_D_CONV, expand=_EXPAND)
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.gate.weight)     # g = 0.5 at init -> 0.5(fwd+bwd)
        nn.init.zeros_(self.gate.bias)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.fwd(h)
        b = self.bwd(h.flip(1)).flip(1)
        g = torch.sigmoid(self.gate(torch.cat([f, b], dim=-1)))   # (B,T,d)
        y = g * f + (1.0 - g) * b
        return x + self.dp(y)


class BiMamba(nn.Module):
    """Stack of gated bidirectional Mamba layers + final RMSNorm."""

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 32):
        super().__init__()
        if len(_DP_MAMBA) != n_layers:
            raise ValueError(
                f"len(_DP_MAMBA)={len(_DP_MAMBA)} != n_layers={n_layers}"
            )
        self.layers = nn.ModuleList([
            BiMambaLayer(d_model, d_state=d_state, drop_path=_DP_MAMBA[i])
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─── Per-subband branch backbone ──────────────────────────────────────────────

class BranchBackbone(nn.Module):
    """TFBlock x3 (dilation 1,2,4) -> flatten F x C -> Linear embed -> BiMamba.

    Input  : (B, d_stem, T2, F2)    Output: (B, T2, d_model)
    """

    def __init__(self, d_stem: int, f2: int, d_model: int = 64,
                 n_mamba_layers: int = 2, d_state: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList([
            TFBlock(d_stem, dilation=_DILATIONS[i], drop_path=_DP_CNN[i])
            for i in range(len(_DILATIONS))
        ])
        in_dim = d_stem * f2
        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            nn.Dropout(_EMBED_DROP),
        )
        self.mamba = BiMamba(d_model, n_layers=n_mamba_layers, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)                                   # (B, d_stem, T2, F2)
        B, C, T, Fd = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fd)  # flatten F x C -> feature
        x = self.embed(x)                                # (B, T2, d_model)
        return self.mamba(x)                             # (B, T2, d_model)


# ─── Adaptive N-way fusion (zero-init -> mean of streams at step 0) ─────────────

class AdaptiveFusion(nn.Module):
    """Merge N branch streams -> (B, T, d). 'gate' = per-channel convex routing.

        cat = concat(S_1 ... S_N)                       (B, T, N·d)
        a   = Linear_{N·d->N·d}(cat) -> (B, T, N, d)
        a   = softmax(a, dim=branch)                    per token, per CHANNEL
        out = sum_i a_i * S_i                            each channel mixes the N
              branches independently. Input-adaptive and convex (a >= 0, sum = 1).

    Zero-init => softmax = 1/N per channel => mean of streams at step 0.
    """

    def __init__(self, d_model: int, n_branches: int):
        super().__init__()
        self.n_branches = n_branches
        self.d_model    = d_model
        self.proj = nn.Linear(n_branches * d_model, n_branches * d_model)
        nn.init.zeros_(self.proj.weight)     # softmax -> 1/N per channel => mean
        nn.init.zeros_(self.proj.bias)

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(streams, dim=-1)                     # (B, T, N·d)
        B, T, _ = cat.shape
        w = self.proj(cat).view(B, T, self.n_branches, self.d_model)
        w = w.softmax(dim=2)                             # (B, T, N, d)
        s = torch.stack(streams, dim=2)                  # (B, T, N, d)
        return (w * s).sum(dim=2)                        # (B, T, d)


# ─── Temporal attentive statistics pooling (ECAPA-style, zero-init) ───────────

class AttnStatPool(nn.Module):
    """Per-channel temporal attention -> [weighted mean || weighted std].

    Input : (B, T, d)    Output: (B, 2*d)

    The score sees [x || mu || sigma] (3 x dim input, full ECAPA style).
    """

    def __init__(self, dim: int):
        super().__init__()
        bn = max(8, dim // 2)
        in_dim = 3 * dim
        self.score = nn.Sequential(
            nn.Linear(in_dim, bn), nn.Tanh(), nn.Linear(bn, dim)
        )
        nn.init.zeros_(self.score[-1].weight)            # uniform attention init
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True).expand_as(x)
        sg = x.var(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6).sqrt().expand_as(x)
        h = torch.cat([x, mu, sg], dim=-1)              # (B, T, 3*d)
        w = self.score(h).softmax(dim=1)                 # (B, T, d)
        mean = (w * x).sum(dim=1)                        # (B, d)
        var = (w * (x - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)


# ─── Classifier head ──────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """LayerNorm -> Dropout -> Linear."""

    def __init__(self, in_dim: int, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(_CLF_DROPOUT),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─── Full model ───────────────────────────────────────────────────────────────

class WavMamba(nn.Module):
    """Multi-branch (per-subband) CNN + gated-BiMamba with adaptive late fusion.

    Only the dataset-dependent dimensions and four width knobs are
    configurable; the architecture flags (subbands, pool, stem_norm, fusion)
    are fixed and any attempt to override them raises ValueError.

    Args:
        num_classes    : output classes (UT-HAR=7, NTU-Fi=6).
        n_antennas     : receive antennas (UT-HAR=3, NTU-Fi=3) = channels per subband.
        f2             : subcarrier axis length after DWT (UT-HAR=15, NTU-Fi=57).
        d_model        : branch feature width (default 64).
        d_stem         : per-subband stem width (default 16).
        d_state        : Mamba SSM state size (default 32).
        n_mamba_layers : BiMamba layers per branch (default 2).

    Input  : X (B, 2*n_antennas, T2, F2), packed [HL | LH] subband-major.
    Output : logits (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int = 7,
        n_antennas: int = 3,
        f2: int = 15,
        d_model: int = 64,
        d_stem: int = 16,
        d_state: int = 32,
        n_mamba_layers: int = 2,
        # Locked architecture — kept as explicit arguments so a stale config
        # fails fast: any deviation from the locked values raises ValueError.
        subbands: tuple = _LOCKED_SUBBANDS,
        pool: str = _LOCKED_POOL,
        stem_norm: bool = _LOCKED_STEM_NORM,
        fusion: str = _LOCKED_FUSION,
    ):
        super().__init__()
        if tuple(subbands) != _LOCKED_SUBBANDS:
            raise ValueError(
                f"WavMamba is fixed to subbands={_LOCKED_SUBBANDS!r}; "
                f"got {tuple(subbands)!r}.")
        if pool != _LOCKED_POOL:
            raise ValueError(f"pool is fixed to {_LOCKED_POOL!r}; got {pool!r}")
        if stem_norm != _LOCKED_STEM_NORM:
            raise ValueError(f"stem_norm is fixed to {_LOCKED_STEM_NORM}; got {stem_norm}")
        if fusion != _LOCKED_FUSION:
            raise ValueError(f"fusion is fixed to {_LOCKED_FUSION!r}; got {fusion!r}")

        self.subbands   = tuple(subbands)
        self.n_branches = len(self.subbands)

        self.f2         = f2                                 # subcarrier axis length
        self.n_per_sub  = n_antennas                         # channels per subband

        # Per-subband stems (always separate — physically-motivated kernels).
        self.stems = nn.ModuleDict({
            s: SubbandStem(self.n_per_sub, d_stem, kernel=_SUBBAND_KERNEL[s])
            for s in self.subbands
        })

        # Separate branch backbones (one per subband).
        self.backbones = nn.ModuleDict({
            s: BranchBackbone(d_stem=d_stem, f2=f2, d_model=d_model,
                              n_mamba_layers=n_mamba_layers, d_state=d_state)
            for s in self.subbands
        })

        self.fusion = AdaptiveFusion(d_model, self.n_branches)
        self.tpool  = AttnStatPool(d_model)
        head_in     = 2 * d_model
        self.head   = Classifier(head_in, num_classes=num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        packed_c = self.n_per_sub * self.n_branches
        if X.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (B, {packed_c}, T2, F2), got {tuple(X.shape)}"
            )
        if X.shape[1] != packed_c:
            raise ValueError(
                f"Expected {packed_c} channels (packed [HL|LH] for subbands "
                f"{self.subbands}), got {X.shape[1]}"
            )
        if X.shape[-1] != self.f2:
            raise ValueError(
                f"Expected F2={self.f2} subcarriers (model built with f2={self.f2}); "
                f"got {X.shape[-1]}."
            )

        # Per-subband stems (separate, physical kernels), then branch backbones.
        streams = []
        for k, s in enumerate(self.subbands):
            sb = X[:, k * self.n_per_sub:(k + 1) * self.n_per_sub]   # (B, M*A, T2, F2)
            stem_out = self.stems[s](sb)                            # (B, d_stem, T2, F2)
            streams.append(self.backbones[s](stem_out))             # (B, T2, d_model)

        z = self.fusion(streams)                                    # (B, T2, d_model)
        z = self.tpool(z)                                           # (B, 2*d_model)
        return self.head(z)                                         # (B, num_classes)
