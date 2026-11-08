#!/usr/bin/env bash
# Hybrid (MonoGS tracker + our mapper) on 600-frame prefixes of three TUM sequences, original mapper vs 3-view batched joint replay.
# About 10-15 min per sequence and condition on the RTX 4050. Results: results/monogs_full/.
cd ~/Documents/external/MonoGS
export PYTHONPATH=$PWD:$HOME/Documents/adaptive_3dgs
for sc in fr1_desk fr3_sitting_static fr1_rpy; do
  echo "== $sc"
  ~/miniconda3/envs/monogs/bin/python ~/Documents/adaptive_3dgs/experiments/run_monogs_hybrid.py \
    --monogs ~/Documents/external/MonoGS --config configs/rgbd/tum/${sc}_local.yaml \
    --conds orig k3_jr3b --n_frames 600 --out_dir ~/Documents/adaptive_3dgs/results/monogs_full 2>&1 | grep --line-buffered -E "DONE|Error|Traceback|out of memory"
done
echo ALLFIN
