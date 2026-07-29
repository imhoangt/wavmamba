# wavmamba — WavMamba for WiFi-CSI HAR (UT-HAR & NTU-Fi)

Public code accompanying the paper. Trains **WavMamba** — a multi-branch
(one branch per Haar DWT subband) CNN + bidirectional-Mamba model with adaptive
late fusion — for WiFi-CSI human activity recognition on **UT-HAR** and
**NTU-Fi**.

The architecture is **fixed** to the configuration reported in the paper:

| Flag | Value |
|------|-------|
| `subbands`   | `('HL', 'LH')`  — Haar 2-branch (no LL) |
| `pool`       | `attnstat`      — attentive statistics pooling |
| `stem_norm`  | `False`         — no GroupNorm in the stem |
| `fusion`     | `gate`          — per-channel softmax branch gate |

Only the dataset-dependent dimensions (`num_classes`, `n_antennas`, `f2`) and
four width knobs (`d_model`, `d_stem`, `d_state`, `n_mamba_layers`) are
configurable; any attempt to change a fixed flag raises `ValueError`.

## Requirements

- Python 3.10+
- CUDA GPU (the Mamba SSM kernels are CUDA-only)
- See `requirements.txt` for the regular Python dependency list:
  - `torch` (2.7+), `numpy`, `scipy`, `pywavelets`, `scikit-learn`, `tqdm`,
    `matplotlib`
- Install `mamba-ssm` and `causal-conv1d` manually as described below because
  their CUDA wheels must match the torch/CUDA/C++ ABI.

## Install

`mamba-ssm` and `causal-conv1d` ship prebuilt CUDA wheels that must match the
torch C++ ABI. With torch 2.7+ from PyPI (`cxx11abi=TRUE`), install the matching
`...cxx11abiTRUE` wheels with `--no-deps` so pip does not re-resolve torch or
replace an ABI-compatible wheel:

```bash
pip install torch==2.7.0
pip install mamba-ssm --no-build-isolation --no-deps
pip install causal-conv1d --no-deps
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` are intentionally **not** listed in
`requirements.txt`; installing them through `pip install -r requirements.txt`
can trigger dependency re-resolution and hard-to-debug CUDA/ABI failures.

> Do **not** build `mamba-ssm` from source unless you have matched the exact
> CUDA/torch ABI — the prebuilt wheels are the reliable path.

## Layout

```
wavmamba/                       repository root
├── wavmamba/                   importable package (the library)
│   ├── __init__.py             public API re-exports
│   ├── __main__.py             command line: python -m wavmamba build | train | ablate
│   ├── model.py                WavMamba (fixed paper configuration)
│   ├── data.py                 raw loaders -> Haar DWT -> bench build -> DataLoaders
│   ├── config.py               TrainCfg — the paper training protocol
│   ├── engine.py               train/eval primitives + analytic MAC counting
│   ├── trainer.py              run() — seed loop -> eval -> plots -> metrics.json
│   └── ablation.py             AblationWavMamba + registry (single-variable study)
├── notebooks/
│   └── wavmamba_kaggle.ipynb   Kaggle companion notebook
├── README.md
├── LICENSE
└── requirements.txt
```

The package holds all logic; `__main__.py` only parses arguments and calls into
it. The CLI examples below assume you run from the repository root:

```bash
cd wavmamba
```

## Datasets

Expected local layout by default:

```
dataset/
├── UT_HAR/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── X_val.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── y_val.csv
└── NTU-Fi_HAR/
    ├── train_amp/
    └── test_amp/
```

You can also keep the data anywhere and pass `--raw-root <path>`; the loaders
recursively search for the expected marker files/folders (`X_train.csv` for
UT-HAR and `train_amp/` for NTU-Fi), which also works with Kaggle dataset mounts.

| Dataset | Classes | n_ant x sub | fs | Split |
|---------|---------|-------------|----|-------|
| UT-HAR  | 7 | 3 x 30  | 100 Hz | official train=X_train / test=X_test+X_val merged (default; `--no-merge-val` keeps test=X_test) |
| NTU-Fi  | 6 | 3 x 114 | 500 Hz | official train_amp / test_amp (time downsampled 2000->500) |

Raw CSI amplitude -> 2-D Haar DWT -> packed `[HL | LH]` subband-major input
`(B, 2*n_antennas, T2, F2)`.

Before public release, replace the dataset placeholders in the Citation /
Provenance section with the official dataset URLs and license/citation notes
required by the dataset providers.

## Normalization — two orthogonal flags

Normalization is split into two independent stages, controlled by `PRENORM`
and `Z_GRAN`. z-norm after the DWT is **always applied**; the flags only choose
the scheme:

| Flag | Values | Meaning |
|------|--------|---------|
| `PRENORM` | `sensefi` \| `none` | Pre-norm on raw amplitude **before** the DWT. `sensefi` = UT-HAR min-max (per split) / NTU-Fi `(x-42.32)/4.98`. `none` = no raw pre-normalization. |
| `Z_GRAN`  | `perpos` \| `pcb`    | Granularity of z-norm **after** the DWT. `perpos` = per-position `(C,T2,F2)`. `pcb` = per-channel-bin `(C,F2)`, collapsing time. |

Four combinations:

| Combination | Meaning |
|-------------|---------|
| `sensefi + pcb`    | raw pre-norm + per-channel-bin z statistics **(default — paper protocol)** |
| `sensefi + perpos` | raw pre-norm + per-position z statistics |
| `none + pcb`       | no raw pre-norm + per-channel-bin z statistics |
| `none + perpos`    | no raw pre-norm + per-position z statistics |

Every distinct build writes to its own bench dir
`bench/<prenorm>_<z_gran>[_mv]/`, so builds never overwrite each other. The
`_mv` suffix marks a UT-HAR merged-val build (the default), which changes the
test split.
`wavmamba/data.py::bench_dirname()` is the single source of truth for that
name and is used by `build_bench()`, the `train` command and the notebook
alike.

**Protocol note.** For exact reproduction of the reported protocol, the DWT
z-normalization statistics are computed over all official split samples in the
bench build (`train + test`, including UT-HAR `val` under the default merged
split). The loader applies those stored statistics to every split. This
all-reps protocol is kept intentionally so the public code matches the reported
preprocessing; it is not presented as a generic train-only normalization
recipe.

## Usage

### CLI (primary entry point)

All commands below assume:

```bash
cd wavmamba
```

Build the bench arrays, then train. The defaults reproduce the paper protocol
(`--prenorm sensefi --z-gran pcb`, UT-HAR test = X_test + X_val, single seed
42), so the bare commands are the reproduction commands:

```bash
# UT-HAR (paper protocol: sensefi pre-norm, per-channel-bin z, merged test split)
python -m wavmamba build --dataset uthar
python -m wavmamba train --dataset uthar

# NTU-Fi (paper protocol)
python -m wavmamba build --dataset ntufi
python -m wavmamba train --dataset ntufi
```

The normalization variants are still available explicitly, e.g. no raw
pre-norm + per-position z:

```bash
python -m wavmamba train --dataset ntufi --prenorm none --z-gran perpos
```

`python -m wavmamba train --help` lists all options (`--seeds`, `--num-epochs`,
`--batch-size`, `--lr`, `--num-workers`, `--raw-root`, `--out-root`,
`--no-merge-val`, `--no-build`, `--bench-dir`). By default `train` builds the
bench first, then trains; pass `--no-build` or `--bench-dir <path>` to reuse an
existing build.

Whenever an existing bench is reused (`--no-build` or `--bench-dir`), the
`train` command reads `stats.json` from that bench and fails if the CLI
dataset/normalization/merge-val labels disagree with the bench metadata, so
results can never be filed under a mismatched tag.

### Python API

The same two steps from Python (useful in notebooks). `build_bench()` defaults
match the CLI (paper protocol):

```python
from wavmamba import build_bench, default_cfg, run

build_bench('uthar', raw_root=RAW, out_root=OUT)
run(bench_dir=BENCH, output_dir=OUT_DIR, cfg=default_cfg())  # single seed 42
```

For the 5-seed statistics reported in the paper, pass
`cfg=default_cfg(seeds=(0, 4, 8, 17, 42))` (CLI: `--seeds 0,4,8,17,42`).

`run()` reads the class count, class names and input dimensions from the bench's
own `stats.json`, so they can never disagree with the data.

Output goes to `<out_root>/outputs/wavmamba_<ds>_<prenorm>_<z_gran>[_mv]/`:

```
metrics.json                     config + per_seed + summary (acc, f1, CM, efficiency)
seeds/<seed:03d>/
    training_log.csv             per-epoch loss / test acc / macro-F1 / timings
    last_model.pt                the reported model (final epoch) — only checkpoint kept
    test_predictions.npz         predictions, probabilities, labels
    training_curve.png           this seed's loss + test-accuracy curve
    confusion_matrix.png         this seed's normalized confusion matrix
plots/                           cross-seed aggregate — only when len(seeds) > 1
    training_curve.png           all seeds overlaid
    confusion_matrix.png         seed-averaged
```

Every seed gets its own figures. `plots/` is the aggregate across seeds, so it is
written only for multi-seed runs (with one seed it would duplicate `seeds/<seed>/`).

Only the final epoch's weights are saved. `per_seed.best_epoch` /
`best_test_acc` in `metrics.json` are train-time diagnostics computed from the
per-epoch test curve; no checkpoint is selected by them, and they must not be
used as headline results.

`summary` reports efficiency at batch size 1: `params_M`, `macs_M`, `flops_M`
(= 2 × MACs), `macs_breakdown_M` per component, and `macs_note` stating the
counting convention. MACs are counted analytically by `wavmamba/engine.py`,
not by a tracer: Mamba's fast path hides its projections and the selective scan
inside a custom autograd Function, where operator-matching counters such as
`fvcore` silently see nothing — about 72% of this model's MACs. `latency_*_ms` is
measured on GPU (200 timed runs after 50 warm-ups) and is `null` on CPU.

`macs_ssm_counted` is `true` when the recurrent sequence core was counted — a
real Mamba block (closed-form scan + projections) or an `nn.LSTM` backbone
(closed-form gates, the `a4_bilstm` ablation). It is `false` only when `mamba_ssm`
is unavailable and a Mamba model falls back to a stand-in, where the number
covers the convolutional path only and must not be reported.

## Ablation study

WavMamba ships a locked architecture, so the ablations live in a separate,
configurable assembler (`wavmamba/ablation.py`) that reuses the same building
blocks under swappable flags — the paper model and its reproducibility guarantee
are untouched. Each variant changes exactly **one** component versus a centre
configuration. `--study a` (the default) centres on the paper model, `ours`:

| # | Component | Variants (`ours` in bold) |
|---|-----------|---------------------------|
| 1 | Front-end       | `a1_raw` (no DWT, raw amplitude) · **dwt (HL+LH)** |
| 2 | Branch structure| `a2_shared` (one shared branch) · **separate per-subband** |
| 3 | CNN stem        | `a3_nostem` (DWT → embed → Mamba) · **stem + 3× TFBlock** |
| 4 | Backbone        | `a4_bilstm` · `a4_unimamba` · **bidirectional Mamba** |
| 5 | fwd/bwd merge   | `a5_add` ((f+b)/2) · `a5_concat` · **per-channel gate** |
| 6 | Branch fusion   | `a6_mean` · `a6_concat` · **adaptive gate** |
| 7 | Pooling         | `a7_mean` · **attentive statistics** |

Run the whole sweep (single seed 42 by default), or a subset:

```bash
python -m wavmamba ablate --dataset uthar                       # all 11 variants
python -m wavmamba ablate --dataset uthar --variants ours,a1_raw,a4_bilstm
python -m wavmamba ablate --dataset uthar --seeds 0,4,8,17,42    # 5-seed table
```

Each variant builds the bench it needs (the DWT bench, or a separate `raw_…`
bench for `a1_raw`), trains under the fixed protocol, and is filed under
`outputs/ablation/<dataset>/<variant>/` with its own `metrics.json`. Finished
variants are skipped, so the sweep is resumable. After the sweep the command
prints a markdown table (acc ± std, macro-F1 ± std, params, MACs, latency per
row); regenerate it any time with `ablation_table('uthar')`, which rediscovers
runs by disk-glob so it survives a kernel restart. No parameter matching is
applied — params and MACs are reported as table columns instead, so the compute
difference of each variant is explicit.

### `--study s`: the WavMamba-S ladder

`--study s` (registry `ABLATIONS_S`, filed under `outputs/ablation_s/`) re-runs
the same seven axes centred on the single-shared-branch variant, reported as the
efficient configuration. Two axes are only expressible from that centre: with one
branch, `c1_raw` differs from the centre by the DWT **alone** (from `ours` it
also changes the branch count), and `c3_split` (per-subband stems, one shared
backbone) separates specialised kernels from a duplicated backbone. `c7_statpool`
drops the pooling attention while keeping the `[mean ‖ std]` statistic, splitting
the two changes that `a7_mean` makes at once.

Three of its rows (`center`, `c1_raw`, `c2_separate`) are configurations study a
already trained, so those runs are reused rather than repeated; a module-level
check in `ablation.py` turns a drifting centre into an `ImportError` instead of a
mislabelled column.

```bash
python -m wavmamba ablate --dataset uthar --study s --seeds 0,4,8,17,42 \
  --variants c3_split,c4_nostem,c5_uni,c5_bilstm,c6_add,c6_concat,c7_statpool,c7_mean
```

### `--study u`: the uni-Mamba cost ladder

`--study u` (registry `ABLATIONS_U`, filed under `outputs/ablation_u/`) centres on
the single-direction backbone. It is a **cost study, not a search for a better
model**: dropping the backward pass removes 39% of the parameters and ~30% of the
latency at accuracy that is statistically indistinguishable from keeping it
(NTU-Fi +0.30pp t=+0.59; UT-HAR −0.10pp t=−1.58 over the same five seeds), so this
ladder asks what the remaining components are worth once the cheap backbone is the
baseline. Read `u5_bimamba` in reverse: here *adding* the backward pass is the
rung.

It has **six axes plus a depth axis**, not seven. With `direction='uni'`,
`_MambaSeqLayer` builds no `bwd` and no gate, so the fwd/bwd merge flag is inert —
`merge='add'` and `merge='concat'` are the same model as the centre, parameter
count and `state_dict` keys identical. The zero-init merge gate can therefore only
be ablated in studies `a` and `s`. In its place, `u7_depth3` moves the backbone
from two Mamba layers to three; drop-path rates come from `_dp_schedule`, which
reproduces both shipped constants exactly (`(0.0, 0.10)` at two layers,
`(0.0, 0.05, 0.10)` at three), so the existing rungs keep their published
regularisation bit-for-bit. Note in any caption that a depth rung necessarily
moves two things at once — layer count *and* the per-layer rates — since the rates
cannot stay fixed when their number changes.

Axis #5 is a full 2×2 over model family × direction, so neither variable moves
alone by accident:

|      | Mamba          | LSTM           |
|------|----------------|----------------|
| uni  | `centre_u`     | `u5_unilstm`   |
| bi   | `u5_bimamba`   | `u5_bilstm`    |

From a uni-Mamba centre, plain `bilstm` would change family *and* direction at
once; `u5_unilstm` (`backbone='unilstm'`) closes that gap. Watch the direction of
the cost, which is the opposite of the Mamba pair: keeping the output at `d_model`
forces a one-directional LSTM to `LSTM(d → d)` where the bidirectional one uses
`LSTM(d → d/2)` per direction, and `W_hh` grows quadratically in the hidden size —
so `u5_unilstm` spends **more** parameters than `u5_bilstm` (+8,192 per layer),
while `unimamba` *saves* 47,040 per layer versus `bimamba`.

Four of its rows (`centre_u`, `u2_separate`, `u5_bimamba`, `u5_bilstm`) are
configurations an earlier study already trained, so those runs are reused rather
than repeated.

```bash
python -m wavmamba ablate --dataset uthar --study u --seeds 0,4,8,17,42 \
  --variants u1_raw,u3_split,u4_nostem,u5_unilstm,u6_statpool,u6_mean,u7_depth3
```

### Reuse across ladders

A re-centred ladder shares rows with the ones before it, and those runs are copied
on disk instead of re-trained — which is only sound while the two configurations
still build the *same model*. That is not the same as carrying the same dict:
several flags are inert depending on the others (`merge` needs a bidirectional
Mamba, `fusion` needs more than one branch, `front_end='raw'` ignores `branch`
entirely), and a later registry carries keys an earlier one lacks
(`n_mamba_layers`). `_canon()` in `wavmamba/ablation.py` reduces a configuration to
what the assembler actually reads, and every reused pair is checked with it at
import time, so a drifting centre becomes an `ImportError` rather than a
mislabelled column. Each copied run directory also carries a `PROVENANCE.txt`
naming its source, since `metrics.json` records no variant name of its own.

### Kaggle notebook (companion)

`notebooks/wavmamba_kaggle.ipynb` runs both datasets end-to-end on Kaggle: clone
the repo, install the dependencies, set `PRENORM`/`Z_GRAN`/`MERGE_VAL`/`SEEDS`,
and run all cells. Before public release, replace `REPO_URL` in the notebook
with the final public repository URL; optionally set `REPO_REF` to the paper
release tag.

## Training protocol (fixed)

`config.py` ships a single protocol:

| | |
|---|---|
| Optimizer  | AdamW, lr=5e-4, betas=(0.9, 0.95), wd=1e-3 |
| Scheduler  | warmup_cosine, warmup=5 epochs, floor_lr=1e-6 |
| Epochs     | 30, batch_size=32, grad_clip=1.0 |
| Loss       | CrossEntropy (fixed — not configurable) |
| WD exclude | norm/bias/A_log/D excluded from weight decay |
| Seeds      | 42 by default; paper 5-seed protocol: (0, 4, 8, 17, 42) |

Seeds are fixed for statistical reproducibility, but training is not bitwise
deterministic by default: cuDNN benchmarking and TF32 matmul are enabled for
speed, and CUDA kernel versions can change exact trajectories.

## Citation / provenance

If you use this code, cite the accompanying WavMamba paper. Replace this block
with the final BibTeX before public release:

```bibtex
@article{wavmamba2026,
  title  = {WavMamba: Wavelet-Guided Bidirectional Mamba for WiFi-CSI Human Activity Recognition},
  author = {<authors>},
  year   = {2026},
  note   = {Code: https://github.com/<owner>/wavmamba}
}
```

Dataset and preprocessing provenance to fill with final official links before
release:

- UT-HAR: add official dataset URL, citation, and license/usage terms.
- NTU-Fi: add official dataset URL, citation, and license/usage terms.
- The raw pre-normalization option `sensefi` follows the public benchmark-style
  preprocessing used for these WiFi-CSI datasets: UT-HAR split-wise min-max and
  NTU-Fi fixed `(x - 42.3199) / 4.9802` scaling.
- `perpos` and `pcb` name the two z-stat granularities used in the experiments;
  the implementation documents their shapes directly so the code remains
  understandable without relying on prior-work labels.

## License

This package is released under the MIT License; see `LICENSE`.
