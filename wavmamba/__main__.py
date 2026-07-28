"""Command-line interface for WavMamba.

    python -m wavmamba build   --dataset uthar
    python -m wavmamba train   --dataset uthar
    python -m wavmamba ablate  --dataset uthar

Defaults reproduce the paper protocol (prenorm=sensefi, z-gran=pcb, UT-HAR
test = X_test + X_val, single seed 42); pass --prenorm/--z-gran/--no-merge-val/
--seeds to deviate.

`build` packs the Haar bench arrays; `train` resolves (and by default builds)
the bench for the requested flags, then runs the multi-seed protocol. `train`
resolves the bench directory with the same bench_dirname() helper the builder
uses, so a run can never be filed under a tag that disagrees with its data.

`ablate` runs the single-variable ablation study: for each variant it builds
the bench that variant needs (the DWT bench, or the raw bench for a1_raw),
trains an AblationWavMamba under the same protocol, and files each result under
outputs/ablation/<dataset>/<variant>/. Finished runs are skipped, so the sweep
is resumable; `ablation_table()` then aggregates them by disk-glob.
"""
import argparse
import json
from pathlib import Path

from .ablation import (
    ABLATIONS, ablation_table, build_ablation_model, variant_front_end,
)
from .config import default_cfg
from .data import DATA_ROOT, DIRMAP, bench_dirname, build_bench
from .trainer import run


def _parse_seeds(s: str) -> tuple:
    """'0,4,8' -> (0, 4, 8)."""
    seeds = tuple(int(t) for t in str(s).replace(' ', '').split(',') if t)
    if not seeds:
        raise argparse.ArgumentTypeError('--seeds must list at least one integer')
    return seeds


def _add_common_args(p):
    p.add_argument('--dataset', required=True, choices=sorted(DIRMAP))
    p.add_argument('--prenorm', default='sensefi', choices=('none', 'sensefi'),
                   help='raw pre-normalization before the DWT (default: sensefi)')
    p.add_argument('--z-gran', default='pcb', choices=('perpos', 'pcb'),
                   help='z-norm granularity after the DWT (default: pcb)')
    p.add_argument('--no-merge-val', dest='merge_val', action='store_false',
                   help='UT-HAR only: keep test=X_test instead of the default '
                        'test=X_test+X_val (paper protocol); no-op for NTU-Fi')
    p.add_argument('--raw-root', default=None,
                   help='raw dataset dir (default: <repo>/../dataset/<dataset dir>)')
    p.add_argument('--out-root', default=None,
                   help='root for bench/ and outputs/ (default: repo dataset/)')


def _check_bench_meta(meta, bench_dir, args, mv):
    """Fail if a reused bench was built with different labels than the CLI asks for."""
    want = {'dataset': args.dataset, 'prenorm': args.prenorm, 'z_gran': args.z_gran}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    # merge_val is None for datasets without a val split; compare as a bool.
    if bool(meta.get('merge_val')) != bool(mv):
        bad['merge_val'] = (bool(meta.get('merge_val')), bool(mv))
    if bad:
        detail = ', '.join(f'{k}: bench={b!r} cli={c!r}' for k, (b, c) in bad.items())
        raise ValueError(
            f'Bench at {bench_dir} does not match the requested run ({detail}). '
            f'Rebuild it or fix the flags — results must not be filed under a '
            f'mismatched tag.')


def _cmd_build(args):
    build_bench(dataset=args.dataset,
                raw_root=args.raw_root,
                out_root=args.out_root,
                merge_val=args.merge_val,
                prenorm=args.prenorm,
                z_gran=args.z_gran)


def _cmd_train(args):
    # merge_val only exists for UT-HAR, so it never taints the NTU-Fi tag.
    mv = args.merge_val and args.dataset == 'uthar'
    out_root = Path(args.out_root) if args.out_root else DATA_ROOT
    tag = bench_dirname(args.prenorm, args.z_gran, mv)

    if args.bench_dir:
        bench_dir = Path(args.bench_dir)
    else:
        bench_dir = out_root / DIRMAP[args.dataset] / 'bench' / tag
        if not args.no_build:
            _cmd_build(args)

    stats_path = bench_dir / 'stats.json'
    if not stats_path.exists():
        raise FileNotFoundError(
            f'No bench at {bench_dir} (stats.json missing). Build it first with '
            f'`python -m wavmamba build` using the same --prenorm/--z-gran/'
            f'--no-merge-val, or drop --no-build/--bench-dir.')
    if args.bench_dir or args.no_build:
        with open(stats_path) as f:
            meta = json.load(f)['meta']
        _check_bench_meta(meta, bench_dir, args, mv)

    overrides = {k: v for k, v in (('num_epochs', args.num_epochs),
                                   ('batch_size', args.batch_size),
                                   ('lr', args.lr)) if v is not None}
    cfg = default_cfg(seeds=args.seeds, **overrides)

    run_name = f'wavmamba_{args.dataset}_{tag}'
    run(bench_dir=bench_dir,
        output_dir=out_root / 'outputs' / run_name,
        cfg=cfg,
        num_workers=args.num_workers)


def _parse_variants(s: str) -> list:
    """'all' -> every registry variant; 'ours,a5_add' -> that subset (order kept)."""
    s = str(s).replace(' ', '')
    if s in ('', 'all'):
        return list(ABLATIONS)
    want = [v for v in s.split(',') if v]
    bad = [v for v in want if v not in ABLATIONS]
    if bad:
        raise argparse.ArgumentTypeError(
            f'unknown variant(s) {bad}; choose from {list(ABLATIONS)} or "all"')
    return want


def _cmd_ablate(args):
    """Run the ablation sweep: build each variant's bench, train, file per variant."""
    out_root = Path(args.out_root) if args.out_root else DATA_ROOT
    overrides = {k: v for k, v in (('num_epochs', args.num_epochs),
                                   ('batch_size', args.batch_size),
                                   ('lr', args.lr)) if v is not None}

    for variant in args.variants:
        fe = variant_front_end(variant)
        # merge_val only exists for UT-HAR; the raw ablation gets the raw bench.
        mv  = args.merge_val and args.dataset == 'uthar'
        tag = bench_dirname(args.prenorm, args.z_gran, mv, fe)
        bench_dir = out_root / DIRMAP[args.dataset] / 'bench' / tag
        out_dir   = out_root / 'outputs' / 'ablation' / args.dataset / variant

        # Resume: skip a variant whose metrics.json already lists every seed.
        metrics_path = out_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                done = json.load(f).get('per_seed', {})
            if all(str(s) in done for s in args.seeds):
                print(f'[skip] {variant}: already complete at {out_dir}')
                continue

        print(f'\n{"#" * 64}\n#  ablate / {args.dataset} / {variant}  ({fe} bench)\n{"#" * 64}')
        if not (bench_dir / 'stats.json').exists():
            build_bench(dataset=args.dataset, raw_root=args.raw_root,
                        out_root=args.out_root, merge_val=args.merge_val,
                        prenorm=args.prenorm, z_gran=args.z_gran, front_end=fe)

        with open(bench_dir / 'stats.json') as f:
            meta = json.load(f)['meta']
        cfg = default_cfg(seeds=args.seeds, **overrides)
        run(bench_dir=bench_dir,
            output_dir=out_dir,
            cfg=cfg,
            num_workers=args.num_workers,
            model_builder=lambda v=variant, m=meta: build_ablation_model(v, m))

    ablation_table(args.dataset, out_root=args.out_root)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='python -m wavmamba',
        description='WavMamba — WiFi-CSI HAR on UT-HAR and NTU-Fi.')
    sub = p.add_subparsers(dest='command', required=True)

    pb = sub.add_parser('build', help='build the packed Haar bench arrays')
    _add_common_args(pb)
    pb.set_defaults(func=_cmd_build)

    pt = sub.add_parser('train', help='build (unless told not to) and train')
    _add_common_args(pt)
    pt.add_argument('--bench-dir', default=None,
                    help='reuse this exact bench dir instead of resolving/building one')
    pt.add_argument('--no-build', action='store_true',
                    help='reuse the resolved bench as-is')
    # Training protocol overrides (everything else is fixed — see config.py).
    pt.add_argument('--seeds', type=_parse_seeds, default=(42,),
                    help='comma-separated seeds (default: 42; paper 5-seed '
                         'protocol: 0,4,8,17,42)')
    pt.add_argument('--num-epochs', type=int, default=None)
    pt.add_argument('--batch-size', type=int, default=None)
    pt.add_argument('--lr', type=float, default=None)
    pt.add_argument('--num-workers', type=int, default=4)
    pt.set_defaults(func=_cmd_train)

    pa = sub.add_parser('ablate', help='run the single-variable ablation sweep')
    _add_common_args(pa)
    pa.add_argument('--variants', type=_parse_variants, default=list(ABLATIONS),
                    help='comma-separated variant names or "all" (default: all). '
                         f'Choices: {", ".join(ABLATIONS)}')
    pa.add_argument('--seeds', type=_parse_seeds, default=(42,),
                    help='comma-separated seeds (default: 42; paper 5-seed '
                         'protocol: 0,4,8,17,42)')
    pa.add_argument('--num-epochs', type=int, default=None)
    pa.add_argument('--batch-size', type=int, default=None)
    pa.add_argument('--lr', type=float, default=None)
    pa.add_argument('--num-workers', type=int, default=4)
    pa.set_defaults(func=_cmd_ablate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
