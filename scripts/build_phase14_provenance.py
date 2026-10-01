#!/usr/bin/env python3
"""Build Phase-14 frozen provenance bundle (auditor P0 requirement).

Reads the merged benchmark JSONs, splits aggregate vs raw, snapshots the
exact pipeline config per policy, hashes every artifact, and writes:
  results/phase14_corrected/{manifest.json, phase14_results.json,
    raw_runs.json, statistics.json, config_snapshot.yaml}

Manifest includes: git SHA + branch + status, dataset paths, scene, frame
range, resolution, seeds, hardware, pose convention, depth-hygiene protocol,
policy configs digest, artifact SHA-256, timestamp.
Reproducible: rerunning regenerates identical hashes given identical inputs.
"""
import sys, json, hashlib, subprocess, platform
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "phase14_corrected"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(cmd):
    try:
        return subprocess.check_output(["git"] + cmd.split(), cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def hardware():
    info = {"platform": platform.platform(), "cpu": platform.processor() or "unknown"}
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
        info["gpu"] = smi
    except Exception:
        info["gpu"] = "unknown (no nvidia-smi)"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def main():
    from experiments.run_phase13_frozen_benchmark import build_pipeline_config
    merged = json.load(open(OUT / "phase14_results.json"))
    raw = merged.pop("raw_runs")
    json.dump(raw, open(OUT / "raw_runs.json", "w"), indent=1)

    meta = merged["metadata"]
    stats = {
        "design": {"n_seeds": 8, "min_wilcoxon_p_two_sided": 2 / 2 ** 8,
                   "correction": "Holm-Bonferroni step-down",
                   "ci": "bootstrap 95% (2000 resamples)"},
        "paired_comparisons": merged.get("paired_comparisons", []),
        "headroom": {
            "formula": "(Q_ours - Q_noop)/(Q_full - Q_noop)",
            "eta": (merged["policy_stats"]["ours"]["mean_psnr"]["mean"]
                    - merged["policy_stats"]["no_op"]["mean_psnr"]["mean"])
                   / (merged["policy_stats"]["full"]["mean_psnr"]["mean"]
                      - merged["policy_stats"]["no_op"]["mean_psnr"]["mean"]),
        },
    }
    json.dump(stats, open(OUT / "statistics.json", "w"), indent=1)

    cfgs = {p: build_pipeline_config(policy=p, seed=42, budget_ms=15.0,
                                     W=320, H=240, device="cuda")
            for p in meta["policies"]}
    import yaml
    yaml.safe_dump({"policies": cfgs,
                    "note": "exact dicts passed as custom_config (seed varies per run)"},
                   open(OUT / "config_snapshot.yaml", "w"), sort_keys=True)

    files = ["phase14_results.json", "raw_runs.json", "statistics.json",
             "config_snapshot.yaml", "README.md",
             "adaptive_office0_150f_8seed.json", "router_heldout.json"]
    raw_status = git("status --short")
    # Files participating in the experiment (code + data + provenance docs).
    # Unrelated worktree changes (e.g. viewer/paper drafts) do not invalidate
    # the frozen experiment, but are listed explicitly instead of claiming a
    # literally clean tree.
    relevant_prefixes = ("research/", "experiments/", "datasets/", "configs/",
                         "tests/", "README.md", "docs/",
                         "results/phase14", "results/rtg_slam", "results/final_confirmation",
                         "results/gate1_headroom", "results/seeds/", "scripts/build_phase14")
    # manifest.json itself is excluded: it records build-time state and can
    # never be clean at its own build moment; its content hashes of the other
    # artifacts are what freeze verification checks.
    relevant_dirty = [ln for ln in raw_status.splitlines()
                      if (ln.strip().split(None, 1)[-1].startswith(relevant_prefixes)
                          and not ln.strip().endswith("results/phase14_corrected/manifest.json"))]
    manifest = {
        "experiment": "phase14_corrected",
        "status": "frozen",
        "freeze_tag": "phase14-corrected-frozen",
        "git": {"sha": git("rev-parse HEAD"), "branch": git("rev-parse --abbrev-ref HEAD"),
                "status_short": raw_status,
                "experiment_relevant_tree_clean": len(relevant_dirty) == 0,
                "unrelated_worktree_changes": [ln for ln in raw_status.splitlines()
                                               if ln not in relevant_dirty]},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"tum_fr2_xyz": "datasets/TUM/rgbd_dataset_freiburg2_xyz",
                    "replica_office0": "datasets/Replica/office0",
                    "association": "nearest-timestamp, max |dt| <= 50 ms"},
        "protocol": {"scene": "tum_fr2_xyz", "n_frames": 150, "resolution": [320, 240],
                     "seeds": [42, 43, 44, 45, 46, 47, 48, 49],
                     "budget_ms": 15.0, "policies": meta["policies"],
                     "pose_convention": "W2C via torch.inverse(TUM C2W groundtruth)",
                     "depth_hygiene": "inpaint small holes (Telea r=4), drop out-of-range [0.4,4]m, "
                                      "dilate Sobel>0.20 edges (3x3) and zero them",
                     "backend": "gsplat (cuda) / reference (cpu)",
                     "tracker": "decoupled (GT poses); no ATE claim"},
        "hardware": hardware(),
        "artifacts": {f: sha256_file(OUT / f) for f in files if (OUT / f).exists()},
        "sources": {
            "rtg_slam_150f_8seeds/final_confirmation_results.json":
                "4 policies x 8 seeds (ours, rtg_slam_reimpl, error_only, error_influence)",
            "phase14_corrected_partial/final_confirmation_results.json":
                "2 policies x 8 seeds (no_op, full)",
        },
        "supersedes": "results/final_confirmation (pose-bugged C2W substrate, see DEPRECATED.md)",
    }
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print("wrote manifest with", len(manifest["artifacts"]), "hashes, git", manifest["git"]["sha"][:8])


if __name__ == "__main__":
    main()
