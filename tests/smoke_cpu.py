"""CPU end-to-end smoke test for the wavmamba package.

Runs the whole pipeline on tiny random arrays shaped like UT-HAR:
    raw .csv (npy-on-disk)  ->  build_bench()  ->  run()  ->  metrics.json

`mamba_ssm` needs CUDA kernels that are not available on a laptop, so
tests/_mock/mamba_ssm.py (Mamba = Linear) is put on sys.path first. This checks
plumbing — shapes, split bookkeeping, artifacts, config surface — not accuracy.

    python tests/smoke_cpu.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / '_mock'))   # mocked mamba_ssm must win over the real one
sys.path.insert(0, str(ROOT))

from wavmamba import bench_dirname, build_bench, default_cfg, run  # noqa: E402

tmp = HERE / '_tmp'
if tmp.exists():
    shutil.rmtree(tmp)
raw = tmp / 'raw'
raw.mkdir(parents=True)

rng = np.random.default_rng(0)


def _save(name, arr):
    """Write an .npy payload under a .csv name — how UT-HAR ships its arrays."""
    np.save(raw / f'{name}.csv', arr, allow_pickle=True)
    (raw / f'{name}.csv.npy').rename(raw / f'{name}.csv')


_save('X_train', rng.normal(10, 3, (48, 250, 90)).astype(np.float32))
_save('y_train', rng.integers(0, 7, 48))
_save('X_test',  rng.normal(10, 3, (16, 250, 90)).astype(np.float32))
_save('y_test',  rng.integers(0, 7, 16))
_save('X_val',   rng.normal(10, 3, (12, 250, 90)).astype(np.float32))
_save('y_val',   rng.integers(0, 7, 12))

out_root = tmp / 'out'

# --- 1) two builds that differ only in --merge-val must not overwrite each other
build_bench('uthar', raw_root=raw, out_root=out_root, merge_val=False,
            prenorm='sensefi', z_gran='perpos')
build_bench('uthar', raw_root=raw, out_root=out_root, merge_val=True,
            prenorm='sensefi', z_gran='perpos')

b_plain = out_root / 'UT_HAR' / 'bench' / bench_dirname('sensefi', 'perpos', False)
b_mv    = out_root / 'UT_HAR' / 'bench' / bench_dirname('sensefi', 'perpos', True)
assert b_plain.exists() and b_mv.exists(), 'bench dirs missing'
n_plain = len(np.load(b_plain / 'y_test.npy'))
n_mv    = len(np.load(b_mv / 'y_test.npy'))
print(f'\nBENCH DIRS  plain={b_plain.name} test={n_plain}  mv={b_mv.name} test={n_mv}')
assert (n_plain, n_mv) == (16, 28), (n_plain, n_mv)   # 16 vs 16+12 merged

meta = json.load(open(b_mv / 'stats.json'))['meta']
print('META KEYS :', sorted(meta))
assert 'n_per_sub' not in meta
assert meta['classes'] == 7 and len(meta['class_names']) == 7
assert meta['merge_val'] is True

# --- 2) train through the library entry point
outdir = tmp / 'run'
run(bench_dir=b_mv, output_dir=outdir,
    cfg=default_cfg(seeds=(0, 1), num_epochs=2, batch_size=8),
    num_workers=0)

arts = sorted(p.relative_to(outdir).as_posix() for p in outdir.rglob('*') if p.is_file())
print(f'\nARTIFACTS ({len(arts)}):')
for a in arts:
    print('  ', a)

mj = json.load(open(outdir / 'metrics.json'))
print('\nmodel_config :', mj['model_config'])
print('config keys  :', len(mj['config']), list(mj['config']))
print('summary keys :', len(mj['summary']))
print('dataset/split:', mj['dataset'], '|', mj['split'])
assert mj['model_config']['n_antennas'] == 3 and mj['model_config']['f2'] == 15
assert mj['dataset'] == 'uthar'
assert 'merged' in mj['split'], mj['split']
assert 'criterion' not in mj['config'], mj['config']
assert 'label_smoothing' not in mj['config']

# --- 3) model_kwargs override (the hook ablations use)
outdir2 = tmp / 'run_override'
run(bench_dir=b_plain, output_dir=outdir2,
    cfg=default_cfg(seeds=(0,), num_epochs=1, batch_size=8),
    num_workers=0, model_kwargs={'d_model': 32})
mj2 = json.load(open(outdir2 / 'metrics.json'))
print('\noverride cfg :', mj2['model_config'])
print('split plain  :', mj2['split'])
assert mj2['model_config']['d_model'] == 32
assert 'merged' not in mj2['split']

shutil.rmtree(tmp)
print('\nSMOKE OK')
