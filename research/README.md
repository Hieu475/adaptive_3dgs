# research/ map

Active code (used by the paper):
- `pipeline.py`, `gaussian_repr.py`, `rasterizer.py`, `densification.py`, `scheduler.py`, `selective_optimizer.py`, `state_store.py`: online RGB-D 3DGS mapper under a modeled per-frame optimization budget.
- `tracker.py` (frame-to-frame odometer), `map_tracker.py` (map-based tracker for the closed-loop check).
- `time_budget.py` (TT), `growth_value.py` + `growth_controller.py` (VGG): the two controllers.
- `holdout_eval.py`, `evaluation.py`, `reproducibility.py`: metrics, held-out views, ATE, provenance.

Earlier line of work, a learned per-Gaussian utility predictor and its protocols (kept for the archived experiments, not used by the paper):
`budget_evaluator`, `budget_selection`, `candidate_context`, `context_*`, `conditional_oracle`, `joint_group_evaluator`, `grouped_batch_sampler`, `scale_invariant_features`, `robust_*`, `normalization_*`, `transfer_protocol`, `closed_loop_*`, `decision_aware_model`, `evidence_protocol`, `utility_*`, `oracle_utility`, `interaction_*`.

Former `phaseN_*` file names are listed in `docs/RENAMES.json` (old name to new name, per directory).
