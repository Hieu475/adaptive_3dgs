# Scripts

Utilities that are not experiments (experiments live in `experiments/`). Run from the repository root with `PYTHONPATH` containing the repo root, e.g. `python scripts/run/run_demo.py`.

| Group | Contents |
|---|---|
| `fetch/` | dataset download/preparation: `download_*`, `upscale_all_replica_scenes.py`, `generate_oracle_dataset.py` |
| `run/` | end-to-end demos and suites: `run_demo.py`, `run_pipeline.py`, `run_all_replica_suite.py`, ablations, `benchmark_multiscene.py`, `reproduce_closed_loop.sh` |
| `viz/` | viewers and figure generators: `view_3d.py`, `visualize_gaussians.py`, `optimize_splat_for_web.py`, `generate_*figures*`, `build_supp_video.py`, `open_web_gpu.sh` |
| `audit/` | provenance and manifest tooling: `audit_experiment_manifests.py`, `verify_*_manifest.py`, `build_generalization_provenance.py`, `generate_state_validation_artifacts.py`, `analyze_interaction_audit.py` |

`docs/layout_renames.json` maps every old flat path to its new location.
