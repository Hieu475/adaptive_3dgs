#!/usr/bin/env bash
# Baselines on the 600-frame prefixes used by run_full_length.sh: original MonoGS (about 15 min per sequence) and SplaTAM (about 1 h per sequence).
# Logs/results: results/baselines/monogs_full/mono_<scene>.log and ~/Documents/external/SplaTAM/experiments/TUM/<scene>_seed0_n600/protocol.json
OUT=~/Documents/adaptive_3dgs/results/baselines/monogs_full; mkdir -p $OUT
cd ~/Documents/external/MonoGS
for sc in fr1_desk fr3_sitting_static fr1_rpy; do
  echo "== MonoGS $sc"
  [ -s $OUT/mono_$sc.log ] && grep -q "Total FPS" $OUT/mono_$sc.log && continue
  MONOGS_MAX_FRAMES=600 MONOGS_REFINE_ITERS=1 ~/miniconda3/envs/monogs/bin/python slam.py --config configs/rgbd/tum/${sc}_local.yaml > $OUT/mono_$sc.log 2>&1
  grep -iE "mean psnr|RMSE ATE|Total FPS|Traceback|out of memory" $OUT/mono_$sc.log | cut -c1-200
done
cd ~/Documents/external/SplaTAM
export PYTHONPATH=$PWD WANDB_MODE=disabled SPLATAM_FRAMES=600
for sc in freiburg1_desk freiburg3_sitting_static freiburg1_rpy; do
  echo "== SplaTAM $sc"; [ -f experiments/TUM/${sc}_seed0_n600/protocol.json ] && continue
  SPLATAM_SCENE=$sc ~/miniconda3/envs/splatam/bin/python scripts/splatam_protocol.py configs/tum/splatam_proto.py 2>&1 | grep --line-buffered -E "PROTOCOL|Error|Traceback|Killed|out of memory|Time Step: (100|300|599)\b"
done
echo ALLFIN
