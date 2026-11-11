#!/usr/bin/env python3
"""Summarize separately preregistered quota-dose studies by scene and frame."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/legacy/phase14_generalization_postfix"
STUDIES = [
    ("replica_office1", 70, "quota_response_frame70"),
    ("replica_office1", 90, "quota_response_frame90"),
    ("tum_fr1_rpy", 70, "quota_response_tum_fr1_rpy_frame70"),
    ("tum_fr1_rpy", 90, "quota_response_tum_fr1_rpy_frame90"),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rows = []
    for scene, frame, folder in STUDIES:
        base = RESULTS / folder
        data_path = base / "analysis.json"
        data = json.loads(data_path.read_text())
        if data["scene"] != scene or data["snapshot_frame"] != frame:
            raise ValueError(f"Study identity mismatch: {folder}")
        if data["valid_seed_snapshots"] != 8:
            raise ValueError(f"Study has incomplete gates: {folder}")
        rows.append({
            "scene": scene,
            "frame": frame,
            "study": folder,
            "quota_mean_gains_db": {q: x["mean_psnr_gain_db"] for q, x in data["quota_curve_zero_spawn"].items()},
            "contrast_75_minus_25": data["primary_contrasts"]["75_minus_25"],
            "contrast_100_minus_96": data["primary_contrasts"]["100_minus_96"],
            "analysis_sha256": sha256(data_path),
            "manifest_sha256": data["provenance"]["manifest_sha256"],
        })
    result = {
        "status": "descriptive synthesis of four distinct preregistered same-snapshot studies",
        "studies": rows,
        "pooled_analysis_performed": False,
        "interpretation": "The response varies with both scene and snapshot frame. Cross-study differences are descriptive; no formal scene-by-frame interaction test or pooled estimate is reported.",
    }
    out_json = RESULTS / "quota_response_scene_stage_matrix.json"
    out_md = RESULTS / "quota_response_scene_stage_matrix.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Mature-quota response by scene and stage",
        "",
        "Four separately preregistered same-snapshot dose-response studies. Each cell uses eight seeds; within-study primary contrasts are paired by seed and Holm-adjusted across two contrasts. The studies are summarized side by side and are not pooled.",
        "",
        "| Scene | Frame | PSNR gain at 25 / 75 / 96 / 100% mature (dB) | 75% − 25% (dB; 95% CI) | Holm p | 100% − 96% (dB; 95% CI) | Holm p | Best among sampled quotas |",
        "|:--|--:|:--|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        g = r["quota_mean_gains_db"]
        a, b = r["contrast_75_minus_25"], r["contrast_100_minus_96"]
        aci, bci = a["seed_bootstrap_95_ci_db"], b["seed_bootstrap_95_ci_db"]
        best = max(g, key=g.get)
        lines.append(
            f"| `{r['scene']}` | {r['frame']} | {g['25']:.3f} / {g['75']:.3f} / {g['96']:.3f} / {g['100']:.3f} | "
            f"{a['mean_effect_db']:+.3f} [{aci[0]:+.3f}, {aci[1]:+.3f}] | {a['holm_adjusted_p']:.4f} | "
            f"{b['mean_effect_db']:+.3f} [{bci[0]:+.3f}, {bci[1]:+.3f}] | {b['holm_adjusted_p']:.4f} | {best}% |"
        )
    lines += [
        "",
        "## What changes across the four studies",
        "",
        "At frame 70, the 75%-versus-25% contrast is negative on Replica office1 and positive on TUM fr1_rpy. At frame 90, the 75% quota beats 25% in both scenes, while the near-full mature contrast (100% versus 96%) is negative on Replica and positive on TUM. The highest mean gain among the four sampled quotas also differs by scene and frame.",
        "",
        "This pattern argues against a single fixed mature quota as a universal rule. It does not identify a deployable scene-state router: only two scenes and two snapshots were measured, under one-frame interventions. Cross-study changes are descriptive, and no pooled p-value or formal interaction test is reported.",
        "",
        "Per-study manifests, preregistrations, seed-level outcomes, and hashes are in each `quota_response_*` directory. The JSON summary records hashes of all four analysis files.",
        "",
    ]
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
