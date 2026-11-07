#!/usr/bin/env bash
# Optimizer-preset ablation at uniform K=10 steps (four main test scenes, two seeds).
set -e
PY="${PYTHON:-python}"; OUT="${1:-results/engine_ablation}"
C="--scenes tum_fr2_xyz tum_fr1_xyz replica_office0 replica_office1 --seeds 42 43 --n_frames 150 --out_dir $OUT --conditions tts --T_ms 1 --n_min 10 --n_max 10 --n_init 10"
$PY experiments/run_vgg.py $C --tag _full
$PY experiments/run_vgg.py $C --preset_off betas --tag _nobetas
$PY experiments/run_vgg.py $C --preset_off lr --tag _nolr
$PY experiments/run_vgg.py $C --preset_off decay --tag _nodecay
$PY experiments/run_vgg.py $C --preset_off lr betas decay --tag _noopt
