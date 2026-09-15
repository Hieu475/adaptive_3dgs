#!/usr/bin/env python3
r"""Phase 10 & 11: Cryptographic Manifest Verification Tool.

Verifies bit-for-bit cryptographic integrity of all generated artifacts against
results/phase10_e2e/manifest.json using SHA-256 digests.

Usage:
    python scripts/verify_phase10_manifest.py [--manifest results/phase10_e2e/manifest.json]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a given file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_manifest(manifest_path: Path) -> bool:
    """Verifies all artifacts listed in the manifest against their on-disk SHA-256."""
    if not manifest_path.exists():
        print(f"[FAIL] Manifest file not found: {manifest_path}", file=sys.stderr)
        return False

    base_dir = manifest_path.parent

    try:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
    except Exception as e:
        print(f"[FAIL] Could not parse manifest JSON: {e}", file=sys.stderr)
        return False

    artifacts = manifest_data.get("artifacts", {})
    if not artifacts:
        print(f"[FAIL] No artifacts declared in manifest: {manifest_path}", file=sys.stderr)
        return False

    print("=" * 80)
    print("   ADAPTIVE 3DGS — CRYPTOGRAPHIC MANIFEST VERIFICATION")
    print("=" * 80)
    print(f">> Manifest:  {manifest_path}")
    print(f">> Phase:     {manifest_data.get('phase', 'Unknown')}")
    print(f">> Artifacts: {len(artifacts)} declared")
    print("-" * 80)
    print(f"{'Artifact Path':<35} {'Size (Bytes)':<14} {'SHA-256 (Prefix)':<18} {'Status'}")
    print("-" * 80)

    all_passed = True
    for rel_path, meta in artifacts.items():
        expected_sha = meta.get("sha256", "")
        expected_size = meta.get("size_bytes", -1)
        file_path = base_dir / rel_path

        if not file_path.exists():
            print(f"{rel_path:<35} {'MISSING':<14} {'N/A':<18} [FAIL: NOT FOUND]")
            all_passed = False
            continue

        actual_size = file_path.stat().st_size
        actual_sha = compute_sha256(file_path)

        if actual_sha == expected_sha and actual_size == expected_size:
            print(f"{rel_path:<35} {actual_size:<14} {actual_sha[:16]}... [PASS]")
        else:
            status_errs = []
            if actual_sha != expected_sha:
                status_errs.append(f"SHA mismatch (exp: {expected_sha[:8]}..., act: {actual_sha[:8]}...)")
            if actual_size != expected_size:
                status_errs.append(f"Size mismatch (exp: {expected_size}, act: {actual_size})")
            print(f"{rel_path:<35} {actual_size:<14} {actual_sha[:16]}... [FAIL: {'; '.join(status_errs)}]")
            all_passed = False

    print("-" * 80)
    if all_passed:
        print(f"   RESULT: ALL {len(artifacts)} ARTIFACTS VERIFIED BIT-FOR-BIT [PASS]")
        print("=" * 80)
        return True
    else:
        print("   RESULT: MANIFEST VERIFICATION FAILED [FAIL]", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify artifacts against manifest.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/phase10_e2e/manifest.json"),
        help="Path to manifest.json (default: results/phase10_e2e/manifest.json)",
    )
    args = parser.parse_args()

    success = verify_manifest(args.manifest)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
