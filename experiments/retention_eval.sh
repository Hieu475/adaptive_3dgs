#!/usr/bin/env bash
# Frozen retention evaluation (final-map PSNR on never-mapped frames) with and without keyframe replay.
set -e
PY="${PYTHON:-python}"; OUT="${1:-results/retention}"
C="--conds orig k3 k3_rp k10 k10_rp --seeds 42 43"
$PY experiments/run_retention.py --scenes tum_fr2_xyz tum_fr1_xyz replica_office0 replica_office1 --n_frames 150 $C --out_dir $OUT/test
$PY experiments/run_retention.py --scenes replica_office3 replica_office4 replica_room0 replica_room2 tum_fr1_room tum_fr1_rpy tum_fr3_sitting_static --n_frames 99 $C --out_dir $OUT/test
for spec in tum_fr2_xyz:300 tum_fr1_xyz:200 tum_fr1_desk:200 tum_fr3_sitting_static:300 tum_fr1_room:200; do
  sc=${spec%%:*}; off=${spec##*:}
  $PY experiments/run_retention.py --scenes $sc --offset $off --n_frames 150 $C --out_dir $OUT/fresh/$sc
done
