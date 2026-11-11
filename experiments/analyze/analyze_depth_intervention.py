#!/usr/bin/env python3
"""Paired policy differences across within-scene depth perturbations."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
base = ROOT / "results/dilution/depth_intervention"
report = {}
for scene_dir in sorted(p for p in base.iterdir() if p.is_dir()):
    records = [json.loads(p.read_text()) for p in scene_dir.glob("*.json")]
    rows = {}
    for r in records:
        rows[(r["level"], r["seed"], r["policy"])] = r
    levels = sorted({k[0] for k in rows})
    report[scene_dir.name] = {}
    for level in levels:
        seeds = sorted({s for (l, s, p) in rows if l == level and
                        (l, s, "error_only") in rows and
                        (l, s, "error_influence_temporal") in rows})
        if not seeds:
            continue
        error = np.asarray([rows[(level, s, "error_only")]["summary"]["mean_psnr"] for s in seeds])
        temporal = np.asarray([rows[(level, s, "error_influence_temporal")]["summary"]["mean_psnr"] for s in seeds])
        delta = temporal - error
        rng = np.random.default_rng(2026)
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
        report[scene_dir.name][str(level)] = {
            "n_seeds": len(seeds), "seeds": seeds,
            "mean_error_psnr": float(error.mean()),
            "mean_temporal_coverage_psnr": float(temporal.mean()),
            "temporal_minus_error_db": float(delta.mean()),
            "paired_seed_ci95_db": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "mean_post_intervention_hole_rate": float(np.mean([
                rows[(level, s, "error_only")]["post_intervention_hole_rate"] for s in seeds])),
        }
out = base / "analysis.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
