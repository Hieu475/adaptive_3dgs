# src/ map

Imported as `from src.<group>.<module> import ...` (the repository root is on `sys.path`, as in every script and test).

- `core/`: online RGB-D 3DGS mapper under a modeled per-frame optimization budget (`pipeline`, `gaussian_repr`, `rasterizer`, `densification`, `scheduler`, `selective_optimizer`, `state_store`, `attribution`, ...).
- `tracking/`: `tracker` (frame-to-frame odometer), `map_tracker` (map-based tracker for the closed-loop check).
- `control/`: `time_budget` (TT), `growth_value` + `growth_controller` (VGG), `noise_adaptive`.
- `evaluation/`: metrics, held-out views and ATE (`holdout_eval`), protocols, benchmark harnesses, provenance (`reproducibility`).
- `mechanism/`: depth/RGB metric bundles and residual selection used by the pipeline. `baselines/`: re-implemented external policy (RTG-SLAM style).
- `utility/`: the earlier learned per-Gaussian utility predictor and its protocols; kept for the archived experiments, not used by the paper.

Old `research/<module>.py` locations and former `phaseN_*` names are listed in `docs/RENAMES.json` (`research_to_src` and per-directory maps).
