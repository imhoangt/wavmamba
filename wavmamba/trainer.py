"""Training driver for WavMamba: seed loop -> eval -> metrics -> plots.

`run()` is the single library entry point; the command-line interface lives in
__main__.py. Training primitives live in engine.py; the plotting and
metrics-serialization helpers live in the Reporting section below and are
callable on their own (e.g. to regenerate a figure from a saved
training_log.csv without re-running training).

    output_dir/
        metrics.json            (config + per_seed + summary)
        plots/                  aggregate over seeds — only when >1 seed
            {training_curve.png, confusion_matrix.png}
        seeds/{seed:03d}/       that seed alone
            {training_log.csv, last_model.pt, test_predictions.npz,
             training_curve.png, confusion_matrix.png}

Only the final epoch's weights are kept (last_model.pt) — that is the reported
model. The best epoch is logged as a diagnostic but never checkpointed.
"""
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from .config import TrainCfg, cfg_asdict, default_cfg
from .data import DISPLAY_NAME, build_loaders, load_stats
from .engine import (
    configure_speed_mode, evaluate, evaluate_full, make_criterion,
    make_optimizer, make_scheduler, measure_efficiency, set_seed, train_epoch,
)
from .model import WavMamba

_COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD']


# ── Reporting: plots ──────────────────────────────────────────────────────────

def _plot_training_curve(log_per_seed: dict, plots_dir: Path, title: str):
    multi      = len(log_per_seed) > 1
    _LOSS_COLOR = '#E74C3C'   # red   — loss always red dashed
    _ACC_COLOR  = '#2ECC71'   # green — acc always green solid (single-seed)
    fig, ax1    = plt.subplots(figsize=(10, 5))
    ax2         = ax1.twinx()

    loss_handles = []
    acc_handles  = []

    for i, (seed, rows) in enumerate(log_per_seed.items()):
        c      = _COLORS[i % len(_COLORS)]
        loss_c = _LOSS_COLOR if not multi else c
        acc_c  = _ACC_COLOR  if not multi else c
        epochs = [r['epoch']         for r in rows]
        losses = [r['train_loss']     for r in rows]
        accs   = [r['test_accuracy'] * 100 for r in rows]
        alpha  = 0.85 if multi else 1.0
        lw     = 1.5  if multi else 2.0
        lbl    = f's={seed}'
        ax1.plot(epochs, losses, color=loss_c, lw=lw, alpha=alpha, ls='--')
        ax2.plot(epochs, accs,   color=acc_c,  lw=lw, alpha=alpha, ls='-')
        loss_handles.append(Line2D([0], [0], color=loss_c, lw=lw, ls='--', label=lbl))
        acc_handles.append( Line2D([0], [0], color=acc_c,  lw=lw, ls='-',  label=lbl))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss',         color=_LOSS_COLOR)
    ax2.set_ylabel('Test Acc (%)', color=_ACC_COLOR)
    ax1.tick_params(axis='y', colors=_LOSS_COLOR)
    ax2.tick_params(axis='y', colors=_ACC_COLOR)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    if not multi:
        loss_handles[0].set_label('Loss')
        acc_handles[0].set_label('Test Acc (%)')
        ax1.legend(handles=[loss_handles[0], acc_handles[0]],
                   loc='center right', fontsize=9)
        fig.tight_layout()
    else:
        loss_hdr = Line2D([], [], color='none', label='Loss')
        acc_hdr  = Line2D([], [], color='none', label='Acc (%)')
        interleaved = [loss_hdr, acc_hdr]
        for lh, ah in zip(loss_handles, acc_handles):
            interleaved.extend([lh, ah])
        fig.legend(handles=interleaved,
                   ncol=len(log_per_seed) + 1,
                   loc='lower center', bbox_to_anchor=(0.5, 0.01),
                   fontsize=8, framealpha=0.95,
                   handlelength=2.5, columnspacing=1.0, handletextpad=0.5)
        fig.subplots_adjust(bottom=0.20, top=0.93, left=0.08, right=0.95)

    fig.savefig(plots_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_confusion_matrix(cms_per_seed: dict, class_names: list,
                            plots_dir: Path, title: str):
    n_cls  = len(class_names)
    cm_avg = np.mean([np.array(c) for c in cms_per_seed.values()], axis=0)
    cm_n   = cm_avg / (cm_avg.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_names, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    n = len(cms_per_seed)
    suffix = f' (avg {n} seeds)' if n > 1 else ''
    ax.set_title(f'{title}{suffix}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f'{cm_n[i, j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if cm_n[i, j] > 0.5 else 'black')
    fig.tight_layout()
    fig.savefig(plots_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Reporting: metrics serialization ──────────────────────────────────────────

def build_metrics(bench_dir, cfg, per_seed_results: dict, summary: dict,
                  model_kwargs: dict = None,
                  dataset: str = None, split: str = None) -> dict:
    """Assemble the full metrics dict from training results."""
    dataset = dataset or 'unknown'
    split   = split   or 'unknown'
    cfg_dict = cfg_asdict(cfg)
    model_config = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in (model_kwargs or {}).items()
    }
    return {
        'model':        'wavmamba',
        'dataset':      dataset,
        'split':        split,
        'eval':         ('Reported metrics (per_seed.test_* and summary.test_*) come '
                         'from last_model.pt = final epoch. The per_seed.best_epoch / '
                         'best_test_acc fields are train-time diagnostics selected by '
                         'peeking at test accuracy and MUST NOT be used as headline results.'),
        'bench_dir':    str(bench_dir) if bench_dir else None,
        'config':       cfg_dict,
        'model_config': model_config,
        'per_seed':     {str(s): v for s, v in per_seed_results.items()},
        'summary':      summary,
    }


def save_metrics(output_dir: Path, metrics: dict):
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


# ── One seed ──────────────────────────────────────────────────────────────────

def _run_seed(seed, cfg, stats, bench_dir, seed_dir, device,
              model_kwargs, num_classes, num_workers, model_builder=None):
    """Train + full-eval one seed.

    Returns (result_dict, log_rows, model). Only the final epoch's weights are
    written (last_model.pt), once, after the loop.

    model_builder: optional callable () -> nn.Module used INSTEAD of the shipped
    WavMamba (the ablation study passes an AblationWavMamba factory here). When
    None — every normal run — the paper model is built exactly as before.
    """
    set_seed(seed)

    train_loader, test_loader = build_loaders(
        stats, bench_dir, batch_size=cfg.batch_size, num_workers=num_workers)

    if model_builder is not None:
        model = model_builder().to(device)
    else:
        model = WavMamba(num_classes=num_classes, **model_kwargs).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    criterion = make_criterion()
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    clip_str = str(cfg.grad_clip) if cfg.grad_clip is not None else 'None'
    print(f'Model    : WavMamba  Device: {device}')
    print(f'Train    : {len(train_loader.dataset)}  Test: {len(test_loader.dataset)}')
    print(f'Params   : {n_params:,} ({n_params / 1e6:.3f}M)')
    print(f'Hyper    : lr={cfg.lr}  bs={cfg.batch_size}  epochs={cfg.num_epochs}  '
          f'wd={cfg.weight_decay}  clip={clip_str}  seeds={list(cfg.seeds)}')
    print('-' * 65)

    log_rows      = []
    t_seed_start  = time.time()
    best_test_acc = -1.0   # < 0 so epoch 1 always sets the diagnostic best
    best_epoch    = 1

    for epoch in range(1, cfg.num_epochs + 1):
        t_ep   = time.time()
        cur_lr = optimizer.param_groups[0]['lr']

        avg_loss, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, cfg.grad_clip)

        ep_time  = time.time() - t_ep
        if epoch < cfg.num_epochs:
            test_acc, test_f1 = evaluate(model, test_loader, device)
        else:
            # Final epoch: one full pass serves both the log row and the report
            # (these weights ARE the reported model — no second eval needed).
            acc, f1, f1_per_cls, cm, preds, probs, gts = evaluate_full(
                model, test_loader, device, num_classes)
            test_acc, test_f1 = acc, f1
        elapsed  = time.time() - t_seed_start

        # Diagnostic only — no checkpoint is written for the best epoch.
        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc
            best_epoch    = epoch

        marker    = '*' if is_best else ' '
        gnorm_tag = '*' if cfg.grad_clip is not None else ''
        print(f'Epoch {epoch:3d}/{cfg.num_epochs}  '
              f'lr={cur_lr:.3e}  loss={avg_loss:.4f}  gnorm={grad_norm:.3f}{gnorm_tag}  |  '
              f'acc={test_acc * 100:.2f}%{marker}  macro_f1={test_f1 * 100:.2f}%  |  '
              f'{ep_time:.1f}s  [{elapsed:.0f}s]')

        log_rows.append({
            'epoch':         epoch,
            'lr':            cur_lr,
            'train_loss':    avg_loss,
            'grad_norm':     round(grad_norm, 6),
            'test_accuracy': test_acc,
            'test_f1_macro': test_f1,
            'epoch_time_s':  round(ep_time, 2),
            'total_time_s':  round(elapsed, 1),
        })

    # Only the final epoch is kept, so save once here — the weights in memory
    # are already the reported ones, no reload needed.
    torch.save(model.state_dict(), seed_dir / 'last_model.pt')

    np.savez(seed_dir / 'test_predictions.npz',
             predictions=np.array(preds, dtype=np.int64),
             probabilities=np.array(probs, dtype=np.float32),
             labels=np.array(gts, dtype=np.int64))

    seed_time = time.time() - t_seed_start
    print(f'Seed {seed} — acc={acc * 100:.2f}%  macro_f1={f1 * 100:.2f}%  '
          f'(best ep={best_epoch} acc={best_test_acc * 100:.2f}%, {seed_time:.0f}s)')

    result = {
        'test_accuracy':         round(acc, 6),
        'test_f1_macro':         round(f1, 6),
        'test_f1_per_class':     [round(v, 6) for v in f1_per_cls],
        'test_confusion_matrix': cm,
        'best_epoch':            best_epoch,
        'best_test_acc':         round(best_test_acc, 6),
        'total_time_s':          round(seed_time),
    }

    with open(seed_dir / 'training_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    return result, log_rows, model


def _summarize(per_seed_results, model, device, meta, total_time):
    """Aggregate per-seed metrics + one-off efficiency measurement."""
    params_m, macs, lat_mean, lat_std = measure_efficiency(
        model, device, (meta['C'], meta['T2'], meta['F2']))

    accs = [v['test_accuracy'] for v in per_seed_results.values()]
    f1s  = [v['test_f1_macro'] for v in per_seed_results.values()]

    return {
        'test_accuracy_mean':  round(float(np.mean(accs)), 6),
        'test_accuracy_std':   round(float(np.std(accs)),  6),
        'test_f1_macro_mean':  round(float(np.mean(f1s)),  6),
        'test_f1_macro_std':   round(float(np.std(f1s)),   6),
        'best_epochs':         [v['best_epoch'] for v in per_seed_results.values()],
        'params_M':            round(params_m, 3),
        # Batch-1 cost, counted analytically (see engine.py, MAC counting), NOT
        # by a tracer: Mamba's fast path hides its projections and the
        # selective scan from operator-level counters. flops_M is 2 x macs_M.
        'macs_M':              macs['total_M'],
        'flops_M':             macs['flops_M'],
        'macs_breakdown_M':    macs['breakdown_M'],
        # False => the run used the CPU test double, so macs_M excludes the SSM.
        'macs_ssm_counted':    macs['ssm_counted'],
        'macs_note':           macs['note'],
        'latency_mean_ms':     lat_mean,
        'latency_std_ms':      lat_std,
        'total_time_s':        total_time,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(bench_dir,
        output_dir,
        cfg: TrainCfg = None,
        num_workers: int = 4,
        model_kwargs: dict = None,
        model_builder=None):
    """Train WavMamba across cfg.seeds and save metrics + plots.

    The dataset labels (classes, class names, split description) are read from
    the bench's own stats.json, so they can never disagree with the data.

    Args:
        bench_dir    : bench/<prenorm>_<z_gran>[_mv]/ built by build_bench()
        output_dir   : where metrics.json / plots / seeds are written
        cfg          : TrainCfg (default: the paper protocol)
        num_workers  : DataLoader workers
        model_kwargs : WavMamba constructor overrides. Defaults to the bench
            dimensions {'n_antennas': meta['n_ant'], 'f2': meta['F2']}; pass
            width knobs (d_model, d_stem, d_state, n_mamba_layers) to override.
            The architecture flags (subbands/pool/stem_norm/fusion) are fixed
            inside the model.
        model_builder: optional callable () -> nn.Module built once per seed
            INSTEAD of the shipped WavMamba (the ablation study passes an
            AblationWavMamba factory here). None — every normal run — builds the
            paper model exactly as before.
    """
    if cfg is None:
        cfg = default_cfg()
    if not cfg.seeds:
        raise ValueError('cfg.seeds is empty — provide at least one seed.')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_speed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats        = load_stats(bench_dir)
    meta         = stats['meta']
    num_classes  = meta['classes']
    class_names  = meta['class_names']
    dataset_name = meta['dataset']
    ds_display   = DISPLAY_NAME.get(dataset_name, dataset_name)
    curve_title  = f'Training Curve of WavMamba on {ds_display} Dataset'
    cm_title     = f'Normalized Confusion Matrix of WavMamba on {ds_display} Dataset'
    model_kwargs = {'n_antennas': meta['n_ant'], 'f2': meta['F2'],
                    **(model_kwargs or {})}

    per_seed_results  = {}
    per_seed_log_rows = {}
    t_total_start     = time.time()

    for si, seed in enumerate(cfg.seeds):
        print(f'\n==== Seed {si + 1}/{len(cfg.seeds)} [seed={seed}] ' + '=' * 38)
        seed_dir = output_dir / 'seeds' / f'{seed:03d}'
        seed_dir.mkdir(parents=True, exist_ok=True)

        result, log_rows, model = _run_seed(
            seed, cfg, stats, bench_dir, seed_dir, device,
            model_kwargs, num_classes, num_workers, model_builder=model_builder)

        per_seed_results[seed]  = result
        per_seed_log_rows[seed] = log_rows

        # This seed's own figures, next to its log and predictions. Same titles
        # as the aggregate — the file location identifies the seed.
        _plot_training_curve({seed: log_rows}, seed_dir, curve_title)
        _plot_confusion_matrix({seed: result['test_confusion_matrix']},
                               class_names, seed_dir, cm_title)

    summary = _summarize(per_seed_results, model, device, meta,
                         round(time.time() - t_total_start))

    if len(cfg.seeds) > 1:
        print(f'\n==== Summary [seeds: {list(cfg.seeds)}] ' + '=' * 33)
        print(f'  acc      = {summary["test_accuracy_mean"] * 100:.2f}%'
              f' +/- {summary["test_accuracy_std"] * 100:.2f}%')
        print(f'  macro_f1 = {summary["test_f1_macro_mean"] * 100:.2f}%'
              f' +/- {summary["test_f1_macro_std"] * 100:.2f}%')
        print(f'  Best epochs : {summary["best_epochs"]}   '
              f'Total: {summary["total_time_s"]}s')
        print('=' * 65)

    # plots/ is the cross-seed aggregate. With a single seed it would just
    # duplicate seeds/<seed>/, so it is only written when there is more than one.
    if len(cfg.seeds) > 1:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_training_curve(per_seed_log_rows, plots_dir, curve_title)
        _plot_confusion_matrix(
            {s: v['test_confusion_matrix'] for s, v in per_seed_results.items()},
            class_names, plots_dir, cm_title)

    metrics = build_metrics(bench_dir, cfg, per_seed_results, summary,
                            model_kwargs=model_kwargs,
                            dataset=dataset_name, split=meta['split'])
    save_metrics(output_dir, metrics)

    print(f'\nSaved : {output_dir}')
