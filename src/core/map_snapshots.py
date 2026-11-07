"""Validation helpers for persisted Gaussian map snapshots."""
import hashlib
from pathlib import Path

import torch


def map_digest_from_snapshot(payload) -> str:
    """Recompute the same digest used by the same-snapshot replay runners."""
    digest = hashlib.sha256()
    for name, tensor in sorted(payload["gaussian_model_state_dict"].items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    for name, value in sorted(payload["gaussian_state_store"].items()):
        if torch.is_tensor(value):
            digest.update(name.encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_map_snapshot(path, *, scene, seed, snapshot_frame, map_digest):
    """Reject a stale, mismatched, or corrupted snapshot at an existing path."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema": "gaussian-map-snapshot-v1",
        "scene": str(scene),
        "seed": int(seed),
        "snapshot_frame": int(snapshot_frame),
        "map_digest": str(map_digest),
    }
    actual = {key: payload.get(key) for key in expected}
    mismatches = {key: {"expected": value, "actual": actual.get(key)}
                  for key, value in expected.items() if actual.get(key) != value}
    computed = map_digest_from_snapshot(payload)
    if computed != map_digest:
        mismatches["payload_map_digest"] = {
            "expected": str(map_digest), "actual": computed}
    if mismatches:
        raise RuntimeError(
            f"Existing map snapshot does not match the current replay prefix: "
            f"{path} ({mismatches}). Rerun with --force to replace the result artifacts."
        )
    return payload
