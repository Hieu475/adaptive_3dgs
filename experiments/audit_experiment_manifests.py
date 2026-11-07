#!/usr/bin/env python3
"""Audit provenance manifests and classify experiment bundles conservatively."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.evaluation.reproducibility import classify_experiment_manifest


DEFAULT_PATHS = [
    "results/final_confirmation",
    "results/phase13_frozen_benchmark",
    "results/phase14_corrected",
]


def audit_path(path: Path) -> Dict[str, Any]:
    if path.is_file():
        manifest_path = path
    else:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() and (path / "provenance.json").is_file():
            manifest_path = path / "provenance.json"
    artifact_dir = manifest_path.parent
    if not manifest_path.is_file():
        if "final_confirmation" in str(path).replace("\\", "/").lower():
            return {
                "path": str(path),
                "status": "legacy",
                "issues": ["Known pre-pose-fix result bundle; no manifest is present."],
                "warnings": ["Do not use its absolute metrics in primary tables."],
            }
        return {
            "path": str(path),
            "status": "invalid",
            "issues": ["manifest.json is missing."],
            "warnings": [],
        }
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(artifact_dir),
            "status": "invalid",
            "issues": [f"Manifest cannot be read: {exc}"],
            "warnings": [],
        }
    if manifest_path.name == "provenance.json":
        run_manifest_path = artifact_dir / "run_manifest.json"
        if run_manifest_path.is_file():
            try:
                with run_manifest_path.open("r", encoding="utf-8") as f:
                    run_manifest = json.load(f)
                for key, value in run_manifest.items():
                    manifest.setdefault(key, value)
            except (OSError, json.JSONDecodeError):
                pass
    result = classify_experiment_manifest(manifest, str(artifact_dir))
    return {"path": str(artifact_dir), **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS,
                        help="Experiment directories or manifest.json paths")
    parser.add_argument("--require-paper-ready", action="store_true",
                        help="Exit nonzero unless every bundle is paper_ready")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Print machine-readable JSON")
    args = parser.parse_args()

    results = [audit_path(Path(p)) for p in args.paths]
    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(f"{result['status'].upper():13} {result['path']}")
            for issue in result.get("issues", []):
                print(f"  ISSUE: {issue}")
            for warning in result.get("warnings", []):
                print(f"  NOTE:  {warning}")

    invalid = any(r["status"] == "invalid" for r in results)
    not_ready = any(r["status"] != "paper_ready" for r in results)
    if invalid or (args.require_paper_ready and not_ready):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
