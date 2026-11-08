#!/usr/bin/env bash
# Runs SplaTAM under our protocol on the six TUM sequences (about 20 min each on the RTX 4050).
cd ~/Documents/external/SplaTAM
export PYTHONPATH=$PWD WANDB_MODE=disabled
for sc in freiburg2_xyz freiburg1_xyz freiburg1_rpy freiburg2_desk freiburg3_sitting_static; do
  echo "== $sc"; [ -f experiments/TUM/${sc}_seed0/protocol.json ] && continue
  SPLATAM_SCENE=$sc ~/miniconda3/envs/splatam/bin/python scripts/splatam_protocol.py configs/tum/splatam_proto.py 2>&1 | grep --line-buffered -E "PROTOCOL|Error|Traceback|Killed|out of memory|Time Step: (50|100|149)\b"
done
echo ALLFIN
