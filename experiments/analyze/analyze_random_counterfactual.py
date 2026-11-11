#!/usr/bin/env python3
"""Uniform-random negative-utility rate on the frozen stratified oracle pool.

For a uniform draw without replacement, the expected negative fraction equals
the pool fraction. Bootstrap resamples seeds, the independent trajectory unit.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
source = ROOT / "results/legacy/gate1_headroom/gate1_summary.json"
rows = json.loads(source.read_text())["seed_records"]
rates = np.array([r["pct_neg"] / 100 for r in rows], dtype=float)
counts = np.array([r["n_visible"] for r in rows], dtype=int)
rng = np.random.default_rng(42)
boot = rates[rng.integers(0, len(rows), (100000, len(rows)))].mean(axis=1)
result = {
    "source": str(source.relative_to(ROOT)),
    "population": "visible geometry-stratified candidates, 60 per seed",
    "selector": "uniform random without replacement",
    "estimand": "expected negative-utility fraction among selected candidates",
    "seed_rates": {str(r["seed"]): float(v) for r, v in zip(rows, rates)},
    "n_candidates": int(counts.sum()),
    "expected_negative_rate": float(rates.mean()),
    "cluster_bootstrap_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
    "limitation": "Does not compare top-ranked error candidates; per-candidate labels were not saved.",
}
out = ROOT / "results/legacy/gate1_headroom/random_selector_negative_rate.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
