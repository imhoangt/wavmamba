"""Data pipeline for WavMamba: raw CSI -> Haar DWT -> bench arrays -> loaders.

The four stages, in flow order (each section below is one stage):

    1. Raw readers + registry   load_uthar / load_ntufi, DATASETS, CLASS_NAMES
    2. Haar DWT front-end       haar_subbands, to_maps
    3. Bench build              build_bench() — packed arrays + stats.json
    4. Torch loading            PreprocWavMambaDataset, build_loaders

Normalization is split into TWO ORTHOGONAL flags:
  prenorm : pre-norm on RAW (BEFORE DWT).
      'sensefi' = UT-HAR min-max/split, NTU-Fi (x-42.32)/4.98.
      'none'    = no raw pre-normalization.
  z_gran  : granularity of z-norm AFTER DWT (applied at load, always on).
      'perpos' = per-position (C,T2,F2).
      'pcb'    = per-channel-bin (C,F2), collapsing time.

Every distinct build writes to its own bench dir so builds never overwrite: the
name carries prenorm, z_gran AND the UT-HAR merge_val flag (which changes the
test split). bench_dirname() is the single source of truth for that name.

Protocol note: stats.json stores all-reps z-normalization statistics computed
over all official split samples in the bench build (train + test, and UT-HAR
val when merge_val=True). This is kept for exact paper-protocol reproduction;
it is not a generic train-only normalization recipe.

Bench layout per (dataset, prenorm, z_gran):

    <out_root>/<DIR>/bench/<prenorm>_<z_gran>[_mv]/
        wavmamba/X_<split>.npy   (N, 2*n_ant, T//2, sub//2) float32, UN-normalized
        y_<split>.npy            (N,)  int64
        stats.json               { 'wavmamba': {mean,std}, 'meta': {...} }
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Default raw dataset root (sibling of this package). Override with --raw-root.
DATA_ROOT = Path(__file__).resolve().parents[2] / 'dataset'

DIRMAP = {'uthar': 'UT_HAR', 'ntufi': 'NTU-Fi_HAR'}

# Human-readable dataset names for figure titles and reports.
DISPLAY_NAME = {'uthar': 'UT-HAR', 'ntufi': 'NTU-Fi'}


# ══ 1. Raw readers + registry ═════════════════════════════════════════════════

def _listing(root, n=50):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file())[:n]


def _find(root: Path, name: str) -> Path:
    """First path under root whose final component == name (file or dir).

    Loaders AUTO-DETECT their files via rglob, so the same code works for the
    local layout and for Kaggle dataset mounts (which nest files under a
    different prefix, e.g. /kaggle/input/ut_har_dataset/data/...).
    """
    hit = next(Path(root).rglob(name), None)
    if hit is None:
        raise FileNotFoundError(f"'{name}' not found under {root}. Files seen: {_listing(root)}")
    return hit


def load_uthar(root, merge_val=False):
    """UT-HAR: .npy stored as .csv. X (N,250,90)=time x (3ant*30sub); needs
    TRANSPOSE -> (N,90,250) feature-major (for DWT). 7 classes.

    merge_val=False: train=X_train(3977), test=X_test(500); val(496) unused.
    merge_val=True : test = X_test + X_val (500+496=996)."""
    def ld(name):
        return np.load(_find(root, f'{name}.csv'), allow_pickle=True)
    Xtr = ld('X_train').astype(np.float32).transpose(0, 2, 1)   # (3977,90,250)
    Xte = ld('X_test').astype(np.float32).transpose(0, 2, 1)    # (500,90,250)
    parts_X = [Xtr, Xte]
    parts_y = [ld('y_train').astype(np.int64), ld('y_test').astype(np.int64)]
    if merge_val:
        parts_X.append(ld('X_val').astype(np.float32).transpose(0, 2, 1))   # (496,90,250)
        parts_y.append(ld('y_val').astype(np.int64))
    X = np.concatenate(parts_X, axis=0)
    y = np.concatenate(parts_y, axis=0)
    n_tr = len(Xtr); n_te = len(X) - n_tr
    splits = {'train': np.arange(n_tr),
              'test':  np.arange(n_tr, n_tr + n_te)}
    return X, y, splits


def load_ntufi(root):
    """NTU-Fi: per-class folders of .mat, key 'CSIamp' (342,2000)=(3ant*114sub) x time.
    Already feature-major (no transpose). Official train_amp/test_amp folders.

    DOWNSAMPLE time x[:, ::4]: 2000 -> 500, matching the public benchmark
    dataloader convention (x = x[:, ::4]; reshape(3,114,500)). The sample is
    500 packets @ 500Hz over 1s."""
    import scipy.io
    train_dir = _find(root, 'train_amp')
    test_dir  = _find(root, 'test_amp')
    classes = sorted(os.listdir(train_dir))   # box,circle,clean,fall,run,walk
    cls2idx = {c: i for i, c in enumerate(classes)}
    Xs, ys, split_idx = [], [], {'train': [], 'test': []}
    k = 0
    for split, folder in [('train', train_dir), ('test', test_dir)]:
        for c in classes:
            cdir = Path(folder) / c
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith('.mat'):
                    continue
                arr = scipy.io.loadmat(str(cdir / fn))['CSIamp'].astype(np.float32)
                if arr.shape != (342, 2000):
                    raise ValueError(
                        f'bad NTU-Fi shape {arr.shape} in {fn}; expected (342, 2000)'
                    )
                arr = arr[:, ::4]                       # 2000 -> 500 (benchmark ::4)
                Xs.append(arr); ys.append(cls2idx[c]); split_idx[split].append(k); k += 1
    X = np.stack(Xs, axis=0)                            # (1200, 342, 500)
    y = np.array(ys, dtype=np.int64)
    splits = {sp: np.array(ix, dtype=np.int64) for sp, ix in split_idx.items()}
    return X, y, splits


DATASETS = {
    # n_ant = receive antennas = channels per subband after packing.
    # Class count is NOT stored here — it is len(CLASS_NAMES[dataset]).
    'uthar': dict(loader=load_uthar, n_ant=3, sub=30),
    'ntufi': dict(loader=load_ntufi, n_ant=3, sub=114),
}

# Class names in LABEL ORDER (0..N-1) for confusion-matrix display.
#   ntufi : sorted train_amp/ folder names (= load_ntufi cls2idx order).
#   uthar : documented UT-HAR benchmark label order; not alphabetical.
CLASS_NAMES = {
    # UT-HAR order checked against public label listings and per-class window counts
    # (walk=label2=most windows, run=label4; transient actions fewer).
    'uthar': ['lie down', 'fall', 'walk', 'pick up', 'run', 'sit down', 'stand up'],
    'ntufi': ['box', 'circle', 'clean', 'fall', 'run', 'walk'],
}

SPLIT_DESC = {
    'uthar': 'official train=X_train; test=X_test (val unused by default)',
    'ntufi': 'official train_amp / test_amp folders',
}

# Optional RAW pre-normalization (applied BEFORE the Haar DWT). Preserves the
# public benchmark convention used by these datasets:
#   uthar : min-max global PER split  (x-min)/(max-min)
#   ntufi : fixed constant (x-42.3199)/4.9802
NTUFI_MEAN, NTUFI_STD = 42.3199, 4.9802


def _sensefi_prenorm(dataset, X, splits):
    """Apply RAW pre-normalization before DWT."""
    if dataset == 'uthar':
        for idx in splits.values():
            seg = X[idx]
            mn, mx = float(seg.min()), float(seg.max())
            X[idx] = (seg - mn) / (mx - mn)
        return X
    if dataset == 'ntufi':
        return ((X - np.float32(NTUFI_MEAN)) / np.float32(NTUFI_STD)).astype(np.float32)
    raise ValueError(f'unknown dataset {dataset!r}')


def load_raw(dataset: str, raw_root, merge_val=False):
    """Dispatch to the dataset's raw loader (merge_val only applies to UT-HAR)."""
    loader = DATASETS[dataset]['loader']
    if dataset == 'uthar':
        return loader(raw_root, merge_val=merge_val)
    return loader(raw_root)


def n_classes(dataset: str) -> int:
    """Class count, derived from the label list so the two can never disagree."""
    return len(CLASS_NAMES[dataset])


def bench_dirname(prenorm: str, z_gran: str, merge_val: bool = False,
                  front_end: str = 'dwt') -> str:
    """Canonical bench sub-directory name.

    merge_val changes the test split, so it MUST be part of the name — otherwise
    a merged-val build and a plain build silently overwrite each other.
    front_end='raw' (the no-DWT ablation input) gets a 'raw_' prefix so it never
    collides with the DWT bench; front_end='dwt' keeps the bare, historical name.
    Single source of truth for build_bench(), the CLI and the notebook.
    """
    prefix = '' if front_end == 'dwt' else f'{front_end}_'
    return f'{prefix}{prenorm}_{z_gran}' + ('_mv' if merge_val else '')


# ══ 2. Haar DWT front-end ═════════════════════════════════════════════════════
# The 2-D Haar DWT splits a sample into (LL, HL, LH, HH) subbands; WavMamba uses
# only the two detail subbands {HL, LH}.
#
# Subband convention:
#     cA, (cH, cV, cD) = pywt.dwt2(flat, 'haar', 'periodization')
#         HL = cV.T   (paper XH — detail along subcarrier axis)
#         LH = cH.T   (paper XV — detail along time axis)
#     (cA = LL approximation and cD = HH diagonal detail are dropped.)
# Packed channel order is canonical [HL | LH], each n_per_sub maps, so
# WavMamba's per-subband stem kernels {HL:(3,7), LH:(7,3)} line up.
#
# Per-sample input to the transform is the merged amplitude `flat`:
#     (n_ant * sub, time)   antenna-major, subcarrier-minor
# Haar is 2-tap and each antenna has an EVEN subcarrier count, so dwt2 on the
# merged axis never mixes antennas — identical to a per-antenna dwt2.

def haar_subbands(flat, n_ant, sub):
    """(n_ant*sub, time) amplitude -> (HL, LH), each (time//2, n_ant*sub//2)."""
    flat = np.asarray(flat, dtype=np.float32)
    expected = n_ant * sub
    if flat.shape[0] != expected:
        raise ValueError(
            f"flat axis0 {flat.shape[0]} != n_ant*sub {expected} "
            f"({n_ant} antennas x {sub} subcarriers)"
        )

    _cA, (cH, cV, _cD) = pywt.dwt2(flat, 'haar', mode='periodization')
    HL = cV.T.astype(np.float32)   # (time//2, n_ant*sub//2)
    LH = cH.T.astype(np.float32)
    return HL, LH


def to_maps(a, n_per_sub):
    """(T, n_per_sub*f2) -> (n_per_sub, T, f2). Unflatten link-major feature axis."""
    T, M = a.shape
    f2 = M // n_per_sub
    return a.reshape(T, n_per_sub, f2).transpose(1, 0, 2)


# ══ 3. Bench build ════════════════════════════════════════════════════════════

def _finalize(s, s2, n):
    """all-reps mean/std from running sums; std floored at 1e-6."""
    mean = s / n
    std = np.maximum(np.sqrt(np.maximum(s2 / n - mean * mean, 0.0)), 1e-6)
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def _raw_map(flat, n_ant, sub):
    """(n_ant*sub, time) amplitude -> (n_ant, time, sub). No DWT.

    The raw ablation front-end: unflatten the antenna-major feature axis and put
    it in the same (channel, T, F) layout the DWT path produces, so the rest of
    the pipeline (packing, z-norm, loaders) is identical apart from the shapes.
    """
    flat = np.asarray(flat, dtype=np.float32)
    expected = n_ant * sub
    if flat.shape[0] != expected:
        raise ValueError(
            f"flat axis0 {flat.shape[0]} != n_ant*sub {expected} "
            f"({n_ant} antennas x {sub} subcarriers)"
        )
    T = flat.shape[1]
    return flat.reshape(n_ant, sub, T).transpose(0, 2, 1)   # (n_ant, T, sub)


def build_bench(dataset: str, raw_root=None, out_root=None,
                merge_val=True, prenorm='sensefi', z_gran='pcb',
                front_end='dwt'):
    """Build packed bench arrays for one dataset.

    front_end='dwt' (default): packs the two Haar subbands {HL, LH} used by
    WavMamba, giving (C=2*n_ant, T2=time//2, F2=sub//2).
    front_end='raw' (the no-DWT ablation input): packs the raw amplitude map as
    (C=n_ant, T=time, F=sub), no DWT — used only by the a1_raw ablation.

    Writes UN-NORMALIZED arrays + stats.json; z-norm is applied at load time.
    See the module docstring for the prenorm x z_gran normalization scheme.

    Defaults reproduce the paper protocol: prenorm='sensefi', z_gran='pcb', and
    (UT-HAR only) merge_val=True — test = X_test + X_val, the split used by the
    public benchmark convention. merge_val is ignored for NTU-Fi, which has no
    val split.
    """
    if prenorm not in ('none', 'sensefi'):
        raise ValueError(f"prenorm must be 'none' | 'sensefi', got {prenorm!r}")
    if z_gran not in ('perpos', 'pcb'):
        raise ValueError(f"z_gran must be 'perpos' | 'pcb', got {z_gran!r}")
    if front_end not in ('dwt', 'raw'):
        raise ValueError(f"front_end must be 'dwt' | 'raw', got {front_end!r}")
    wav_subs = ('HL', 'LH')   # fixed — matches WavMamba
    per_channel_bin = (z_gran == 'pcb')   # pcb: z-norm collapse time axis
    cfg = DATASETS[dataset]
    n_ant, sub = cfg['n_ant'], cfg['sub']

    raw_root = Path(raw_root) if raw_root else DATA_ROOT / DIRMAP[dataset]
    base_out = Path(out_root) / DIRMAP[dataset] if out_root else DATA_ROOT / DIRMAP[dataset]

    print(f'Loading {dataset} from {raw_root} ...')
    X, y, splits = load_raw(dataset, raw_root, merge_val=merge_val)
    N, AxS, time = X.shape
    expected_axis = n_ant * sub
    if AxS != expected_axis:
        raise ValueError(
            f'loaded feature axis {AxS} != n_ant*sub {expected_axis} '
            f'({n_ant} antennas x {sub} subcarriers) for {dataset}'
        )
    if prenorm == 'none':
        print(f'  prenorm=none: skip raw pre-normalization for {dataset}')
    else:
        X = _sensefi_prenorm(dataset, X, splits)
        print(f'  prenorm=sensefi: raw pre-normalization applied for {dataset}')

    # Output map dims depend on the front-end: DWT halves both axes and doubles
    # the channel count (one map per subband); raw keeps time/sub and one map
    # per antenna.
    if front_end == 'dwt':
        C, Tdim, Fdim = len(wav_subs) * n_ant, time // 2, sub // 2
        print(f'  N={N}  raw=({AxS},{time})  wav_subs={wav_subs}  -> wav=({C},{Tdim},{Fdim})')
    else:
        C, Tdim, Fdim = n_ant, time, sub
        print(f'  N={N}  raw=({AxS},{time})  front_end=raw  -> map=({C},{Tdim},{Fdim})')
    print(f'  classes={n_classes(dataset)}  prenorm={prenorm}  z_gran={z_gran}')
    print('  split: ' + '  '.join(f'{k}={len(v)}' for k, v in splits.items()))

    # front_end x prenorm x z_gran x merge_val -> one bench dir each, never
    # overwrite. merge_val only exists for UT-HAR (the only split with a val
    # set), so it never taints the NTU-Fi dir name.
    bench_mv = merge_val and dataset == 'uthar'
    out_dir = base_out / 'bench' / bench_dirname(prenorm, z_gran, bench_mv, front_end)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp, idx in splits.items():
        np.save(out_dir / f'y_{sp}.npy', y[idx].astype(np.int64))

    (out_dir / 'wavmamba').mkdir(exist_ok=True)
    mm = {sp: np.lib.format.open_memmap(str(out_dir / 'wavmamba' / f'X_{sp}.npy'),
          mode='w+', dtype=np.float32, shape=(len(idx), C, Tdim, Fdim))
          for sp, idx in splits.items()}
    # z_gran='perpos': per-position stats (C,Tdim,Fdim) — one mean per position.
    # z_gran='pcb':    per-channel-bin stats (C,Fdim), collapsing time.
    _wsh = (C, Fdim) if per_channel_bin else (C, Tdim, Fdim)
    s = np.zeros(_wsh); s2 = np.zeros(_wsh); n_wav = np.int64(0)

    for sp, idx in splits.items():
        for j, i in enumerate(tqdm(idx, desc=f'  [{sp}]', unit='smp')):
            if front_end == 'dwt':
                HL, LH = haar_subbands(X[i], n_ant, sub)
                _sb = {'HL': HL, 'LH': LH}                       # pack wav_subs, in order
                x = np.concatenate([to_maps(_sb[w], n_ant) for w in wav_subs],
                                   axis=0).astype(np.float32, copy=False)
            else:
                x = _raw_map(X[i], n_ant, sub).astype(np.float32, copy=False)
            mm[sp][j] = x
            xd = x.astype(np.float64)
            if per_channel_bin:                       # collapse time -> (C,Fdim)
                s += xd.sum(axis=1); s2 += (xd * xd).sum(axis=1); n_wav += Tdim
            else:                                     # per-position (C,Tdim,Fdim)
                s += xd; s2 += xd * xd; n_wav += 1

    meta = dict(dataset=dataset,
                n_ant=n_ant, sub=sub, C=C, T2=Tdim, F2=Fdim,
                front_end=front_end,       # 'dwt' | 'raw' — input map type
                classes=n_classes(dataset), class_names=CLASS_NAMES[dataset],
                split=('official train=X_train; test=X_test+X_val (merged)'
                       if (dataset == 'uthar' and merge_val) else SPLIT_DESC[dataset]),
                merge_val=(merge_val if dataset == 'uthar' else None),
                subband_order=('|'.join(wav_subs) if front_end == 'dwt' else 'raw'),
                norm='all-reps',
                prenorm=prenorm,           # 'none' | 'sensefi' — pre-norm raw (before DWT)
                z_gran=z_gran)             # 'perpos' | 'pcb' — granularity z-norm after DWT
    stats = {'meta': meta}
    mean, std = _finalize(s, s2, n_wav)            # pcb:(C,Fdim) | perpos:(C,Tdim,Fdim)
    stats['wavmamba'] = {'mean': mean, 'std': std}
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f)
    for sp in list(mm):
        del mm[sp]
    print(f'  saved -> {out_dir}  (y_*, stats.json)  un-normalized')


# ══ 4. Torch loading ══════════════════════════════════════════════════════════

def load_stats(bench_dir) -> dict:
    with open(Path(bench_dir) / 'stats.json') as f:
        return json.load(f)


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _kw(num_workers):
    kw = dict(pin_memory=True, num_workers=num_workers,
              persistent_workers=(num_workers > 0))
    if num_workers > 0:
        kw['worker_init_fn'] = _worker_init_fn
    return kw


class PreprocWavMambaDataset(Dataset):
    """Loads wavmamba/X_<split>.npy, applies z-norm, returns (X, label).

    z-norm is ALWAYS applied. The mean/std shape is either:
      (C, F2)   per-channel-bin  [z_gran=pcb]   -> broadcast to (C, 1, F2)
      (C, T2,F2) per-position    [z_gran=perpos] -> used as-is
    """

    def __init__(self, bench_dir: Path, split: str, stats: dict):
        bench_dir = Path(bench_dir)
        self.X   = np.load(bench_dir / 'wavmamba' / f'X_{split}.npy', mmap_mode='r')
        self.y   = np.load(bench_dir / f'y_{split}.npy')
        s = stats['wavmamba']
        def _bcast(a):
            a = np.array(a, dtype=np.float32)
            return a[:, None, :] if a.ndim == 2 else a
        self.mu = _bcast(s['mean']); self.sig = _bcast(s['std'])

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        x = np.array(self.X[idx], dtype=np.float32)                 # copy -> writable
        x = (x - self.mu) / self.sig          # mu/sig broadcast (C,1,F2) | (C,T2,F2)
        return torch.from_numpy(x), int(self.y[idx])


def build_loaders(stats: dict, bench_dir, batch_size: int = 32, num_workers: int = 4):
    """Build (train_loader, test_loader) from pre-built bench arrays.

    Args:
        stats      : dict from load_stats(bench_dir)
        bench_dir  : path to bench/<prenorm>_<z_gran>[_mv]/
    """
    bench_dir = Path(bench_dir)
    sentinel  = bench_dir / 'wavmamba' / 'X_train.npy'
    if not sentinel.exists():
        raise FileNotFoundError(
            f'Bench arrays not found: {sentinel}\n'
            'Run `python -m wavmamba build` first.')

    print(f'  Loaded: {bench_dir}')

    kw = _kw(num_workers)
    train_ds = PreprocWavMambaDataset(bench_dir, 'train', stats)
    test_ds  = PreprocWavMambaDataset(bench_dir, 'test',  stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader
