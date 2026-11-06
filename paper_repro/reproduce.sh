#!/usr/bin/env bash
# One command per paper table. See paper_repro/README.md.
# Usage: bash paper_repro/reproduce.sh {list|check|main|factorial|depth|adaptive|holdout|tracking|figs}
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cmd="${1:-list}"
case "$cmd" in
  list)
    sed -n '2,3p' "$0"; echo "targets: check main factorial depth adaptive holdout tracking figs" ;;
  check)
    $PY paper_repro/check_numbers.py ;;
  main)   # Tab. 1 (clean 7-policy bundle).
    if [ -d results/phase14_projected_area_fix_clean_7policy_8seeds ] && [ "${FORCE:-0}" != "1" ]; then
      echo "Frozen bundle exists (results/phase14_projected_area_fix_clean_7policy_8seeds); refusing to regenerate."
      echo "Move it aside and re-run, or set FORCE=1 to rerun the six-policy and RTG runs (the merge step still refuses to overwrite)."
      exit 3
    fi
    $PY experiments/run_final_confirmation.py --scene tum_fr2_xyz --n_frames 150 --seeds 42 43 44 45 46 47 48 49 \
        --out_dir results/phase14_projected_area_fix_clean_8seeds
    $PY experiments/run_final_confirmation.py --scene tum_fr2_xyz --n_frames 150 --seeds 42 43 44 45 46 47 48 49 \
        --policies rtg_slam_reimpl --out_dir results/phase14_projected_area_fix_clean_rtg8
    $PY experiments/merge_phase14_clean_rtg_results.py
    $PY experiments/write_phase14_fix_sensitivity_manifest.py --out_dir results/phase14_projected_area_fix_clean_7policy_8seeds ;;
  factorial)  # Tab. 2
    $PY experiments/run_dilution_factorial.py
    $PY experiments/analyze_dilution_factorial.py ;;
  depth)      # Tab. 4
    $PY experiments/run_depth_intervention.py
    $PY experiments/analyze_depth_intervention.py ;;
  adaptive)   # router, mixed regime
    $PY experiments/run_adaptive_policy.py --scenes tum_fr2_xyz replica_office0 --n_frames 150 --seeds 42 43 44
    $PY experiments/run_mixed_regime.py ;;
  holdout)    # every-8th-frame render-only proxy
    $PY experiments/run_holdout_benchmark.py --scene tum_fr2_xyz --n_frames 30 --seeds 42 43 44 --out_dir results/phase14_holdout ;;
  tracking)   # baseline VO sanity check
    $PY experiments/run_tracking_ate.py --scenes tum_fr2_xyz tum_fr1_xyz --n_frames 100 --seeds 42 43 44 --out_dir results/phase14_tracking100 ;;
  figs)
    $PY docs/paper/gen_iccv_figs.py ;;
  *) echo "unknown target: $cmd"; exit 2 ;;
esac
