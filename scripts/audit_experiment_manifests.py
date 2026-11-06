#!/usr/bin/env python3
"""Audit result manifests for provenance completeness and known pose validity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("commit", "config_hash", "dataset", "pose_convention", "seed", "backend")


def first(data, *keys):
    for key in keys:
        value = data.get(key)
        if value not in (None, "", "none", "unknown"):
            return value
    return None


def audit(path: Path):
    data = json.loads(path.read_text())
    git = data.get("git") if isinstance(data.get("git"), dict) else {}
    protocol = data.get("protocol") if isinstance(data.get("protocol"), dict) else {}
    missing = []
    artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    config_artifact_hash = artifacts.get("config_snapshot.yaml")
    values = {
        "commit": first(data, "git_sha", "git_commit") or first(git, "sha"),
        "config_hash": first(data, "config_sha256", "config_hash") or config_artifact_hash,
        "dataset": first(data, "dataset_name", "scene", "dataset") or first(protocol, "scene"),
        "pose_convention": first(data, "pose_convention") or first(protocol, "pose_convention"),
        "seed": first(data, "seed", "seeds", "seed_list") or first(protocol, "seeds"),
        "backend": first(data, "renderer_backend", "backend") or first(protocol, "backend"),
    }
    missing = [key for key in REQUIRED if values[key] is None]
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    config_file = path.parent / "config_snapshot.yaml"
    config_hash_mismatch = False
    if config_artifact_hash and config_file.exists():
        actual_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
        config_hash_mismatch = actual_hash != config_artifact_hash
    commit_sha = str(values["commit"] or "").removesuffix("-dirty")
    commit_valid = bool(re.fullmatch(r"[0-9a-fA-F]{40}", commit_sha))
    if commit_valid:
        commit_valid = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    if config_hash_mismatch or (values["commit"] and not commit_valid):
        status, reason = "invalid", "manifest commit or config snapshot hash failed validation"
    elif rel.startswith("results/final_confirmation/"):
        status = "invalid"
        reason = "known pre-pose-fix results; excluded from primary tables"
    elif rel.startswith("results/phase14_corrected/") and data.get("status") == "frozen":
        frozen_commit = not str(values["commit"]).endswith("-dirty")
        status = "paper-ready" if not missing and frozen_commit and not config_hash_mismatch else "exploratory"
        reason = "frozen corrected-pose evidence" if status == "paper-ready" else "required provenance fields missing or worktree was dirty"
    elif "/_archive/" in rel or rel.startswith("results/_archive/"):
        status, reason = "legacy", "archived artifact"
    elif missing or str(values["commit"]).endswith("-dirty") or values["config_hash"] == "none":
        status, reason = "exploratory", "incomplete provenance or dirty/configless run"
    else:
        status, reason = "exploratory", "requires protocol-specific scientific review"
    return {"manifest": rel, "status": status, "reason": reason,
            "missing_fields": missing, "commit_exists": commit_valid,
            "config_snapshot_hash_mismatch": config_hash_mismatch, **values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="manifest files or result directories (default: results)")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "manifest_audit.json")
    args = parser.parse_args()
    candidates = []
    roots = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.paths] or [ROOT / "results"]
    for item in roots:
        if item.is_file() and item.name == "manifest.json":
            candidates.append(item)
        elif item.is_dir():
            candidates.extend(item.rglob("manifest.json"))
    rows = [audit(p) for p in sorted(set(candidates))]
    counts = {label: sum(row["status"] == label for row in rows)
              for label in ("paper-ready", "exploratory", "legacy", "invalid")}
    report = {"audit_version": "1.0", "counts": counts, "manifests": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(counts, sort_keys=True))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
