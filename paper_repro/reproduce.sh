#!/usr/bin/env bash
# One command per paper table. See paper_repro/README.md.
# Usage: bash paper_repro/reproduce.sh {list|check|main|factorial|depth|adaptive|scenes|counterfactual|extension|dose|scenes_budget|long|ablation|holdout_ext|tables|holdout|tracking|figs}
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cmd="${1:-list}"
case "$cmd" in
  list)
    sed -n '2,3p' "$0"; echo "targets: check main factorial depth adaptive scenes counterfactual extension dose scenes_budget long ablation holdout_ext tables holdout tracking figs" ;;
  check)
    $PY paper_repro/check_numbers.py ;;
  main)   # Tab. 1 + Tab. 2 + Fig. 1: traceable rerun from a tagged commit (writes results/rerun_paper1/).
    # Check out the tag first:  git checkout paper-rerun-1   (worktree must be clean)
    if [ -d results/rerun_paper1/main ] && [ "${FORCE:-0}" != "1" ]; then
      echo "results/rerun_paper1/main exists; refusing to overwrite. Move it aside or set FORCE=1."; exit 3
    fi
    mkdir -p results/rerun_paper1
    printf "commit: %s\ntag: paper-rerun-1\nworktree_clean_at_start: %s\n" "$(git rev-parse HEAD)" \
      "$( [ -z "$(git status --porcelain)" ] && echo yes || echo no)" > results/rerun_paper1/SOURCE.txt
    $PY experiments/run_final_confirmation.py --scene tum_fr2_xyz --n_frames 150 --seeds 42 43 44 45 46 47 48 49 \
        --policies no_op error_only error_influence error_influence_temporal rtg_slam_reimpl ours full \
        --out_dir results/rerun_paper1/main
    $PY experiments/run_dilution_factorial.py --out_dir results/rerun_paper1/dilution_factorial
    $PY experiments/analyze_dilution_factorial.py --input results/rerun_paper1/dilution_factorial ;;
  factorial)  # Tab. 2
    $PY experiments/run_dilution_factorial.py
    $PY experiments/analyze_dilution_factorial.py ;;
  depth)      # Tab. 4
    $PY experiments/run_depth_intervention.py
    $PY experiments/analyze_depth_intervention.py ;;
  adaptive)   # router, mixed regime
    $PY experiments/run_adaptive_policy.py --scenes tum_fr2_xyz replica_office0 --n_frames 150 --seeds 42 43 44
    $PY experiments/run_mixed_regime.py ;;
  scenes)     # per-scene breakdown / confirmatory comparisons: ours vs error_only
    # name:scene:frames:seeds   (seeds exactly as recorded in each bundle's metadata)
    for spec in fr1_desk:tum_fr1_desk:150:"42 43 44" fr1_room:tum_fr1_room:100:"42 43 44" fr1_rpy:tum_fr1_rpy:100:"42 43 44" \
                fr1_xyz:tum_fr1_xyz:150:"42 43 44" fr1_xyz_b:tum_fr1_xyz:150:"45 46 47 48 49" \
                fr3_static:tum_fr3_sitting_static:100:"42 43 44" \
                office1:replica_office1:100:"42 43 44" office1_b:replica_office1:100:"45 46 47 48 49" \
                office2:replica_office2:99:"42 43 44" office3:replica_office3:99:"42 43 44" office4:replica_office4:99:"42 43 44" \
                room0:replica_room0:99:"42 43 44" room1:replica_room1:100:"42 43 44" room2:replica_room2:99:"42 43 44"; do
      IFS=: read -r name scene frames seeds <<<"$spec"
      if [ -d "results/phase14_multiscene150/$name" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "skip $name (exists; FORCE=1 to overwrite)"; continue
      fi
      $PY experiments/run_final_confirmation.py --scene "$scene" --n_frames "$frames" --seeds $seeds \
          --policies ours error_only --out_dir "results/phase14_multiscene150/$name"
    done
    echo "NOTE: the 8-seed tables (fr1_xyz_8seed.json, office1_8seed.json) merge the two seed batches; the merge script is not recorded." ;;
  counterfactual)  # negative-utility strata (gate-1) and the random-selector reference rate
    $PY experiments/run_gate1_headroom.py
    $PY experiments/analyze_random_counterfactual.py ;;
  extension)  # Replica factorial + map-proportional budget (+ TUM unthrottled/proportional pair)
    $PY experiments/run_dilution_extension.py --scenes replica_office0 replica_office1
    $PY experiments/run_dilution_extension.py --scenes tum_fr2_xyz tum_fr1_xyz --conditions error_off error_budget_prop
    $PY experiments/analyze_dilution_extension.py ;;
  dose)       # budget dose-response, free vs capped map (caps: 23k TUM, 60k office0, 40k office1)
    $PY experiments/run_dilution_dose.py --scenes tum_fr2_xyz tum_fr1_xyz --cap 23000
    $PY experiments/run_dilution_dose.py --scenes replica_office0 --cap 60000 --out_dir results/dilution_dose_replica_cap/office0_cap60k
    $PY experiments/run_dilution_dose.py --scenes replica_office1 --cap 40000 --out_dir results/dilution_dose_replica_cap/office1_cap40k
    $PY experiments/analyze_dilution_dose.py ;;
  scenes_budget)  # 11 further scenes (99 frames, 3 seeds): unthrottled / coverage / proportional budget
    $PY experiments/run_dilution_extension.py --scenes replica_office2 replica_office3 replica_office4 replica_room0 replica_room1 replica_room2 \
        tum_fr1_desk tum_fr1_room tum_fr1_rpy tum_fr3_sitting_static tum_fr3_structure_texture_near \
        --conditions error_off error_coverage error_budget_prop --seeds 42 43 44 --n_frames 99 --out_dir results/dilution_scenes
    $PY experiments/analyze_dilution_scenes.py ;;
  long)       # 500-frame TUM sequences, 2 seeds (Replica copies are only 100-120 frames)
    $PY experiments/run_dilution_extension.py --scenes tum_fr2_xyz tum_fr1_xyz tum_fr1_desk \
        --conditions error_off error_coverage error_budget_prop --seeds 42 43 --n_frames 500 --out_dir results/dilution_long
    $PY experiments/analyze_dilution_long.py ;;
  ablation)   # decomposition of coverage throttling + simpler growth controls (4 scenes x 4 seeds)
    $PY experiments/run_throttle_ablation.py
    $PY experiments/analyze_throttle_ablation.py ;;
  holdout_ext)  # held-out-view check of budget and throttling controls (4 scenes x 4 seeds)
    $PY experiments/run_holdout_extension.py
    $PY experiments/analyze_holdout_extension.py ;;
  tables)     # regenerate LaTeX tables, prose and macros from frozen JSON (CPU)
    $PY paper_repro/make_main_tables.py
    $PY paper_repro/make_tables.py ;;
  holdout)    # every-8th-frame render-only proxy
    $PY experiments/run_holdout_benchmark.py --scene tum_fr2_xyz --n_frames 30 --seeds 42 43 44 --out_dir results/phase14_holdout ;;
  tracking)   # baseline VO sanity check
    $PY experiments/run_tracking_ate.py --scenes tum_fr2_xyz tum_fr1_xyz --n_frames 100 --seeds 42 43 44 --out_dir results/phase14_tracking100 ;;
  figs)
    $PY docs/paper/gen_iccv_figs.py
    $PY paper_repro/make_fig_dilution.py ;;
  *) echo "unknown target: $cmd"; exit 2 ;;
esac
