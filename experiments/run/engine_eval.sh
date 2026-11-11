#!/usr/bin/env bash
# Frozen evaluation of the steps regime (fast core, tuned optimizer, time-targeted step count).
# usage: bash experiments/run/engine_eval.sh <out_root> ; resumable (finished runs are skipped).
set -e
PY="${PYTHON:-python}"; OUT="${1:-results/controllers/engine}"
run_unit() {  # scene n_frames offset out_dir
  sc=$1; nf=$2; off=$3; od=$4
  C="--scenes $sc --seeds 42 43 --n_frames $nf --offset $off --out_dir $od"
  $PY experiments/run/run_vgg.py $C --conditions error_off tt_vgg error_off_opt tt_vgg_opt
  for K in 3 6 10 15; do
    $PY experiments/run/run_vgg.py $C --conditions tts --T_ms 1 --n_min $K --n_max $K --n_init $K --tag _fixK$K
  done
  for T in 45 60 75 90; do $PY experiments/run/run_vgg.py $C --conditions tts_vgg --T_ms $T --tag _T$T; done
  $PY experiments/run/run_vgg.py $C --conditions tts --T_ms 60 --tag _T60
}
for sc in tum_fr2_xyz tum_fr1_xyz replica_office0 replica_office1; do run_unit $sc 150 0 $OUT/test/$sc; done
for sc in replica_office3 replica_office4 replica_room0 replica_room2 tum_fr1_room tum_fr1_rpy tum_fr3_sitting_static; do run_unit $sc 99 0 $OUT/test/$sc; done
for spec in tum_fr2_xyz:300 tum_fr1_xyz:200 tum_fr1_desk:200 tum_fr3_sitting_static:300 tum_fr1_room:200; do
  sc=${spec%%:*}; off=${spec##*:}; run_unit $sc 150 $off $OUT/fresh/$sc
done
