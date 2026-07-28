"""CPU smoke test for the ablation study surface.

Verifies, under the same mocked-`mamba_ssm` harness as smoke_cpu.py:
  1. every registry variant builds + forwards to (B, num_classes) on a synthetic
     UT-HAR bench (the raw variant on the raw bench, the rest on the DWT bench);
  2. ABLATIONS['ours'] reproduces the shipped WavMamba layer-for-layer
     (identical parameter count) — the assembler cannot drift from the paper model;
  3. the a4_bilstm MAC count includes the fused-LSTM term (the fvcore trap);
  4. `python -m wavmamba ablate` runs end-to-end and ablation_table() aggregates.

    python tests/smoke_ablation.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / '_mock'))   # mocked mamba_ssm must win over the real one
sys.path.insert(0, str(ROOT))

from wavmamba import (  # noqa: E402
    ABLATIONS, AblationWavMamba, WavMamba, ablation_table, build_ablation_model,
    build_bench,
)
from wavmamba.data import load_stats                       # noqa: E402
from wavmamba.engine import wavmamba_macs                  # noqa: E402

tmp = HERE / '_tmp_abl'
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

# --- build both benches the ablation needs (DWT for 10 variants, raw for a1_raw)
build_bench('uthar', raw_root=raw, out_root=out_root,
            merge_val=True, prenorm='sensefi', z_gran='pcb', front_end='dwt')
build_bench('uthar', raw_root=raw, out_root=out_root,
            merge_val=True, prenorm='sensefi', z_gran='pcb', front_end='raw')

b_dwt = out_root / 'UT_HAR' / 'bench' / 'sensefi_pcb_mv'
b_raw = out_root / 'UT_HAR' / 'bench' / 'raw_sensefi_pcb_mv'
assert b_dwt.exists() and b_raw.exists(), 'ablation bench dirs missing'
meta_dwt = load_stats(b_dwt)['meta']
meta_raw = load_stats(b_raw)['meta']
print(f'DWT bench C,T,F = {meta_dwt["C"]},{meta_dwt["T2"]},{meta_dwt["F2"]}  '
      f'front_end={meta_dwt["front_end"]}')
print(f'RAW bench C,T,F = {meta_raw["C"]},{meta_raw["T2"]},{meta_raw["F2"]}  '
      f'front_end={meta_raw["front_end"]}')
assert (meta_dwt['C'], meta_dwt['T2'], meta_dwt['F2']) == (6, 125, 15)
assert (meta_raw['C'], meta_raw['T2'], meta_raw['F2']) == (3, 250, 30)
assert meta_raw['front_end'] == 'raw' and meta_dwt['front_end'] == 'dwt'

# --- 1) every variant builds + forwards to (B, num_classes) with head_in per pool
print(f'\nFORWARD all {len(ABLATIONS)} variants:')
for variant in ABLATIONS:
    meta = meta_raw if ABLATIONS[variant]['kwargs']['front_end'] == 'raw' else meta_dwt
    model = build_ablation_model(variant, meta).eval()
    C, T, F = meta['C'], meta['T2'], meta['F2']
    with torch.no_grad():
        out = model(torch.randn(2, C, T, F))
    assert out.shape == (2, 7), f'{variant}: bad output {tuple(out.shape)}'
    pool = ABLATIONS[variant]['kwargs']['pool']
    head_in = model.head.net[-1].in_features
    d_model = 64
    assert head_in == (2 * d_model if pool == 'attnstat' else d_model), \
        f'{variant}: head_in {head_in} wrong for pool={pool}'
    n_par = sum(p.numel() for p in model.parameters())
    print(f'  {variant:12s} -> {tuple(out.shape)}  head_in={head_in}  '
          f'params={n_par:,}')

# --- 2) ours reproduces the shipped WavMamba layer-for-layer (same param count)
ours = AblationWavMamba(**ABLATIONS['ours']['kwargs'],
                        num_classes=7, n_antennas=3, f2=15)
paper = WavMamba(num_classes=7, n_antennas=3, f2=15)
p_ours  = sum(p.numel() for p in ours.parameters())
p_paper = sum(p.numel() for p in paper.parameters())
print(f'\nOURS vs WavMamba params : {p_ours:,} vs {p_paper:,}')
assert p_ours == p_paper, f'assembler drifted: {p_ours} != {p_paper}'

# --- 3) a4_bilstm MACs include the fused-LSTM term (not silently dropped)
bilstm = build_ablation_model('a4_bilstm', meta_dwt)
macs = wavmamba_macs(bilstm, (meta_dwt['C'], meta_dwt['T2'], meta_dwt['F2']))
print(f'\na4_bilstm MACs breakdown: {macs["breakdown_M"]}')
assert macs['breakdown_M'].get('lstm', 0) > 0, 'LSTM MACs missing (fused-module trap)'
assert macs['ssm_counted'] is True, 'BiLSTM seq core should count as counted'
# hand cross-check one direction of one layer: 4*(in+h)*h*L*dirs, h=d_model//2=32
h, d_model, L, dirs = 32, 64, 125, 2
per_layer = L * dirs * 4 * (d_model + h) * h        # layer 0: in_size = d_model
per_layer2 = L * dirs * 4 * (h * dirs + h) * h      # layer 1: in_size = h*dirs
# two branches (HL, LH), each 2-layer BiLSTM
expect_lstm = 2 * (per_layer + per_layer2)
got_lstm = round(macs['breakdown_M']['lstm'] * 1e6)
print(f'  lstm MACs expect={expect_lstm:,}  got={got_lstm:,}')
assert got_lstm == expect_lstm, (got_lstm, expect_lstm)

# --- 4) end-to-end via the CLI: two variants, one seed, then aggregate
from wavmamba.__main__ import main  # noqa: E402

main(['ablate', '--dataset', 'uthar', '--variants', 'ours,a5_add', '--seeds', '0',
      '--num-epochs', '1', '--batch-size', '8', '--num-workers', '0',
      '--raw-root', str(raw), '--out-root', str(out_root)])

abl_root = out_root / 'outputs' / 'ablation' / 'uthar'
for v in ('ours', 'a5_add'):
    mp = abl_root / v / 'metrics.json'
    assert mp.exists(), f'missing {mp}'
    s = json.load(open(mp))['summary']
    assert 'params_M' in s and 'macs_M' in s
    print(f'\n{v}: acc={s["test_accuracy_mean"]:.3f} params={s["params_M"]}M '
          f'macs={s["macs_M"]}M')

print('\nABLATION TABLE:')
table = ablation_table('uthar', out_root=str(out_root))
assert '| ours |' in table and '| a5_add |' in table

# ablation_table must also persist the summary to disk (the paper numbers).
abl_dir = out_root / 'outputs' / 'ablation' / 'uthar'
assert (abl_dir / 'summary.md').exists(),  'summary.md not written'
assert (abl_dir / 'summary.csv').exists(), 'summary.csv not written'
csv_head = (abl_dir / 'summary.csv').read_text().splitlines()[0]
assert csv_head == 'variant,acc,f1_macro,params_M,macs_M,ssm_counted,note', csv_head

shutil.rmtree(tmp)
print('\nABLATION SMOKE OK')
