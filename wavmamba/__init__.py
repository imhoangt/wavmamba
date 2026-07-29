"""WavMamba — wavelet multi-branch Mamba for WiFi-CSI human activity recognition.

Modules
-------
model      the WavMamba network (fixed paper configuration)
data       raw loaders -> Haar DWT -> bench build -> torch DataLoaders
config     TrainCfg — the training protocol
engine     train/eval primitives + analytic MAC counting + efficiency probe
trainer    run() — multi-seed training driver, plots, metrics.json
ablation   AblationWavMamba + registry for the single-variable ablation study
__main__   command line: python -m wavmamba build | train | ablate

Typical use
-----------
    from wavmamba import build_bench, run, default_cfg

    build_bench('uthar')   # defaults = the paper protocol (sensefi, pcb, merged val)
    run(bench_dir='.../bench/sensefi_pcb_mv', output_dir='.../outputs/uthar',
        cfg=default_cfg(num_epochs=100))
"""

from .config import TrainCfg, default_cfg
from .data import (
    CLASS_NAMES,
    DIRMAP,
    bench_dirname,
    build_bench,
    build_loaders,
    haar_subbands,
    load_stats,
    to_maps,
)
from .model import WavMamba
from .trainer import run
from .ablation import (
    ABLATIONS,
    ABLATIONS_S,
    AblationWavMamba,
    ablation_table,
    build_ablation_model,
    variant_front_end,
)

__version__ = '1.0.0'

__all__ = [
    'ABLATIONS', 'ABLATIONS_S', 'AblationWavMamba', 'CLASS_NAMES', 'DIRMAP',
    'TrainCfg', 'WavMamba', 'ablation_table', 'bench_dirname',
    'build_ablation_model', 'build_bench', 'build_loaders', 'default_cfg',
    'haar_subbands', 'load_stats', 'run', 'to_maps', 'variant_front_end',
]
