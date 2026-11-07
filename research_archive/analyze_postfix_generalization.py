#!/usr/bin/env python3
"""Summarize the clean post-projected-area-fix held-out scene reruns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/phase14_generalization_postfix"
SOURCE_COMMIT = "75865c7921b5db9d5e8998fc94957935c798f1d6"
SOURCE_HASHES = {
    "experiments/run_final_confirmation.py": "ad685fca7dcced1278579497800d96f23917185dd4c3911f8324f231709483f6",
    "experiments/run_frozen_benchmark.py": "ced3f339df2c77537b7c0238bb965f7e8907aeede2c15edfc19b150d171beb44",
    "research/attribution.py": "e415f83276d5938b41770888dbe559af575c141ef477288bf15ef263cada7af8",
    "research/pipeline.py": "7f2fc76a883150136ff23b7ecd483b920c2d70ba6463ac325765d85f993db6da",
    "research/scheduler.py": "93a87f7b471f3b561e13ebb9962402631fc3cc49f6252631f40b792584a4b5d3",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(scene: str) -> dict:
    path = OUT / scene / "final_confirmation_results.json"
    return json.loads(path.read_text())


def main() -> None:
    scenes = {
        "replica_office1": "replica_office1_150f",
        "tum_fr1_rpy": "tum_fr1_rpy_150f",
    }
    records = {name: load(folder) for name, folder in scenes.items()}
    lines = [
        "# Post-fix held-out generalization check",
        "",
        "Clean-source paired reruns comparing `ours` with `error_only`, 8 paired seeds (42–49), 320×240, ground-truth poses, gsplat, modeled optimization budget 15 ms. The budget is not end-to-end latency.",
        "",
        "| Scene | Frames actually loaded | OURS mean PSNR | Error-only mean PSNR | Paired delta (OURS − error-only) | OURS final PSNR | Error-only final PSNR | OURS / error-only Gaussians |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    scene_rows = {}
    for scene, data in records.items():
        stats = data["policy_stats"]
        ours = stats["ours"]
        err = stats["error_only"]
        delta = next(c for c in data["paired_comparisons"] if c["baseline"] == "error_only")
        nframes = 120 if scene.startswith("replica_") else 151
        lines.append(
            f"| `{scene}` | {nframes} | {ours['mean_psnr']['mean']:.2f} ± {ours['mean_psnr']['std']:.2f} | "
            f"{err['mean_psnr']['mean']:.2f} ± {err['mean_psnr']['std']:.2f} | "
            f"{delta['delta_mean_psnr']:+.2f} dB (Wilcoxon p={delta['p_mean_psnr']:.4f}) | "
            f"{ours['final_psnr']['mean']:.2f} | {err['final_psnr']['mean']:.2f} | "
            f"{ours['N_final']['mean']:.0f} / {err['N_final']['mean']:.0f} |"
        )
        scene_rows[scene] = {
            "requested_frames": 150,
            "actual_frames_loaded": nframes,
            "ours_mean_psnr_db": ours["mean_psnr"]["mean"],
            "error_only_mean_psnr_db": err["mean_psnr"]["mean"],
            "paired_delta_ours_minus_error_only_db": delta["delta_mean_psnr"],
            "paired_wilcoxon_p": delta["p_mean_psnr"],
            "ours_final_psnr_db": ours["final_psnr"]["mean"],
            "error_only_final_psnr_db": err["final_psnr"]["mean"],
            "ours_mean_final_gaussians": ours["N_final"]["mean"],
            "error_only_mean_final_gaussians": err["N_final"]["mean"],
        }

    lines += [
        "",
        "## Reading the result",
        "",
        "The post-fix result still shows scene-dependent behavior: `ours` loses mean PSNR on both held-out scenes, with a larger loss on Replica office1. On TUM fr1_rpy it has higher final-frame PSNR than `error_only` despite lower trajectory-mean PSNR, so a single endpoint does not represent the full trajectory. The smaller Gaussian population is a resource tradeoff, not evidence of better reconstruction quality on these scenes.",
        "",
        "Replica office1 contains only 120 frames in the available dataset although 150 were requested; the runner loads the available prefix. TUM fr1_rpy loads all 151 frames (the runner requests `n_frames + 1`). Results are paired by seed; the frame measurements are not treated as independent samples.",
        "",
        "This is a two-scene diagnostic rerun, not a broad generalization claim. The scenes were chosen as known regression cases after earlier cross-scene results, and only two policies were compared. Do not pool these values with older dirty or pre-fix artifacts.",
        "",
        f"Source commit: `{SOURCE_COMMIT}` (clean frozen snapshot at run time). Renderer: gsplat. Device: RTX 4050 Laptop GPU. The source and per-run artifacts are SHA-256 recorded in `manifest.json`.",
        "",
    ]
    report = OUT / "generalization_report.md"
    report.write_text("\n".join(lines))

    artifacts = {}
    for folder in scenes.values():
        for filename in ("final_confirmation_results.json", "final_confirmation_report.md"):
            path = OUT / folder / filename
            artifacts[str(path.relative_to(OUT))] = sha256(path)
    artifacts["generalization_report.md"] = sha256(report)
    manifest = {
        "experiment": "phase14_postfix_heldout_generalization",
        "status": "exploratory",
        "protocol_version": "phase14-postfix-paired-heldout-v1",
        "git_sha": SOURCE_COMMIT,
        "source_status_at_run_time": "clean frozen source snapshot",
        "source_sha256": SOURCE_HASHES,
        "dataset": "Replica office1 and TUM freiburg1_rpy",
        "renderer_backend": "gsplat",
        "pose_convention": "ground-truth W2C via Phase 10 loader; mapping-only",
        "protocol": {
            "requested_frames": 150,
            "actual_frames_loaded": {k: (120 if k.startswith("replica_") else 151) for k in records},
            "resolution_wh": [320, 240],
            "seeds": list(range(42, 50)),
            "policies": ["ours", "error_only"],
            "budget_ms": 15.0,
            "budget_scope": "modeled optimization budget, not end-to-end latency",
            "statistical_unit": "complete trajectory per seed",
            "commands": [
                "python experiments/run_final_confirmation.py --scene replica_office1 --n_frames 150 --budget_ms 15 --seeds 42 43 44 45 46 47 48 49 --policies ours error_only --out_dir results/phase14_generalization_postfix/replica_office1_150f --device cuda",
                "python experiments/run_final_confirmation.py --scene tum_fr1_rpy --n_frames 150 --budget_ms 15 --seeds 42 43 44 45 46 47 48 49 --policies ours error_only --out_dir results/phase14_generalization_postfix/tum_fr1_rpy_150f --device cuda",
            ],
        },
        "hardware": {
            "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "python": "3.12.13",
            "pytorch": "2.11.0+cu128",
            "cuda": "12.8",
            "os": "Linux-6.8.0-94-generic-x86_64-with-glibc2.39",
        },
        "scene_results": scene_rows,
        "artifacts_sha256": artifacts,
        "limitations": [
            "Only two known regression scenes were rerun; this is diagnostic, not a held-out generalization study designed before policy development.",
            "Replica office1 provides only 120 frames despite a request for 150.",
            "Only OURS and error_only were compared.",
            "The 15 ms figure is a modeled optimization budget; measured end-to-end FPS is reported in per-scene result files.",
            "SciPy emitted a NumPy version compatibility warning during the run; benchmark completed and produced finite metrics.",
        ],
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
