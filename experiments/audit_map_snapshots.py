#!/usr/bin/env python3
"""Audit saved Gaussian map snapshots against experiment records.

Checks file SHA256 and the in-memory map digest recorded by the factorial
runners. This validates artifact integrity; it does not claim a snapshot is a
complete pipeline checkpoint (optimizer/tracker/cache state are not included).
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_map_digest(payload) -> str:
    digest = hashlib.sha256()
    model = payload["gaussian_model_state_dict"]
    for name, tensor in sorted(model.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    store = payload["gaussian_state_store"]
    for name, value in sorted(store.items()):
        if torch.is_tensor(value):
            digest.update(name.encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def resolve_snapshot(record_path: Path, relative_path: str):
    for parent in record_path.parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?",
                        default=Path("results/phase14_generalization_postfix/mechanism_growth_starvation"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for record_path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        snapshot_rel = record.get("map_snapshot_path")
        if not snapshot_rel:
            continue
        snapshot_path = resolve_snapshot(record_path, snapshot_rel)
        row = {
            "record": str(record_path),
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "expected_file_sha256": record.get("map_snapshot_sha256"),
            "expected_map_digest": record.get("base_snapshot_hash"),
            "status": "missing" if snapshot_path is None else "unchecked",
        }
        if snapshot_path is not None:
            row["actual_file_sha256"] = sha256_file(snapshot_path)
            row["file_sha256_match"] = (
                row["expected_file_sha256"] == row["actual_file_sha256"]
                if row["expected_file_sha256"] else None
            )
            try:
                payload = torch.load(snapshot_path, map_location="cpu", weights_only=True)
                row["payload_schema"] = payload.get("schema")
                row["snapshot_map_digest"] = payload.get("map_digest")
                row["computed_map_digest"] = payload_map_digest(payload)
                row["map_digest_match_record"] = (
                    row["computed_map_digest"] == row["expected_map_digest"]
                    if row["expected_map_digest"] else None
                )
                row["map_digest_match_snapshot"] = (
                    row["computed_map_digest"] == row["snapshot_map_digest"]
                    if row["snapshot_map_digest"] else None
                )
                checks = [row["file_sha256_match"], row["map_digest_match_record"],
                          row["map_digest_match_snapshot"]]
                row["status"] = "verified" if all(x is True for x in checks) else "mismatch_or_incomplete"
            except Exception as exc:  # preserve audit of malformed/old artifacts
                row["status"] = "unreadable"
                row["error"] = f"{type(exc).__name__}: {exc}"
        records.append(row)

    counts = {}
    for row in records:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "schema": "gaussian-map-snapshot-audit-v1",
        "root": str(root),
        "records_with_snapshots": len(records),
        "status_counts": counts,
        "scope_note": "Integrity only; these map snapshots do not store all pipeline state and are not restartable full-pipeline checkpoints.",
        "records": records,
    }
    out = args.out or (root / "snapshot_integrity_audit.json")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    md = out.with_suffix(".md")
    lines = [
        "# Gaussian map snapshot integrity audit", "",
        f"Root: `{root}`", "",
        f"Snapshot references checked: **{len(records)}**", "",
        "| Status | Count |", "|:--|--:|",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines += ["", report["scope_note"], ""]
    if counts.get("mismatch_or_incomplete") or counts.get("missing") or counts.get("unreadable"):
        lines += ["## Non-verified records", ""]
        for row in records:
            if row["status"] != "verified":
                lines.append(f"- `{row['record']}`: **{row['status']}** — {row.get('error', row.get('snapshot_path'))}")
    md.write_text("\n".join(lines))
    print(md)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
