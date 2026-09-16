#!/usr/bin/env python3
r"""Phase 12: Manifest Verification Script.

Verifies bit-for-bit SHA-256 hashes and file sizes for all Phase 12 Paper Evidence artifacts.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "results" / "phase12_paper_evidence" / "manifest.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not MANIFEST_PATH.exists():
        print(f"FAIL: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    base_dir = MANIFEST_PATH.parent
    all_passed = True
    print("=" * 80)
    print("   VERIFYING PHASE 12 PAPER EVIDENCE ARTIFACTS")
    print("=" * 80)

    for rel_path, expected in manifest.get("artifacts", {}).items():
        p = base_dir / rel_path
        if not p.exists():
            print(f"[MISSING] {rel_path}")
            all_passed = False
            continue

        actual_sha = compute_sha256(p)
        actual_size = p.stat().st_size

        if actual_sha != expected["sha256"]:
            print(f"[SHA MISMATCH] {rel_path}: expected {expected['sha256'][:16]}..., got {actual_sha[:16]}...")
            all_passed = False
        elif actual_size != expected["size_bytes"]:
            print(f"[SIZE MISMATCH] {rel_path}: expected {expected['size_bytes']}, got {actual_size}")
            all_passed = False
        else:
            print(f"[PASS] {rel_path} ({actual_size} bytes, SHA: {actual_sha[:16]}...)")

    print("-" * 80)
    if all_passed:
        print(">> ALL PHASE 12 ARTIFACTS VERIFIED BIT-FOR-BIT!")
        sys.exit(0)
    else:
        print(">> VERIFICATION FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
