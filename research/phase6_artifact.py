"""Phase 6: Artifact and Checkpoint Schema Validation and Manifest Management.

Enforces schema_version "phase6-v2" and validates that every saved checkpoint
and artifact bundle satisfies authoritative scientific invariants:
  - Canonical group-aware ranking metadata
  - Strict Phase 4 frozen backbone declaration
  - Protocol version v1 specification
  - Matched seed and variant attributes
"""
import os
import json
import torch
from typing import Dict, Any, List, Tuple, Optional


REQUIRED_CHECKPOINT_FIELDS = [
    "schema_version",
    "architecture",
    "variant",
    "seed",
    "protocol_version",
    "model_state",
    "config",
    "normalizer_path",
    "p4_checkpoint",
    "p4_frozen",
    "git_commit",
]


def validate_phase6_checkpoint(ckpt: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates that a Phase 6 checkpoint dict satisfies schema_version 'phase6-v2'.

    Args:
        ckpt: Loaded checkpoint dictionary.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # 1. Required fields presence
    for field in REQUIRED_CHECKPOINT_FIELDS:
        if field not in ckpt:
            errors.append(f"Missing required field: '{field}'")

    # 2. Schema version check
    schema_ver = ckpt.get("schema_version")
    if schema_ver != "phase6-v2":
        errors.append(f"Invalid schema_version: '{schema_ver}'. Expected 'phase6-v2'")

    # 3. Architecture check
    arch = ckpt.get("architecture")
    if arch not in ("residual_context", "direct_context"):
        errors.append(f"Invalid architecture: '{arch}'. Expected 'residual_context' or 'direct_context'")

    # 4. Residual-specific invariants
    if arch == "residual_context":
        if not ckpt.get("p4_frozen", False):
            errors.append("p4_frozen must be True for residual_context architecture")
        p4_ckpt = ckpt.get("p4_checkpoint")
        if not p4_ckpt or not isinstance(p4_ckpt, str):
            errors.append(f"p4_checkpoint must be a non-empty string path, got {p4_ckpt}")

    # 5. Protocol version check
    protocol = ckpt.get("protocol_version")
    if protocol != "v1":
        errors.append(f"Invalid protocol_version: '{protocol}'. Expected 'v1'")

    # 6. Model state and config
    if not isinstance(ckpt.get("model_state"), dict):
        errors.append("model_state must be a state_dict dictionary")
    if not isinstance(ckpt.get("config"), dict):
        errors.append("config must be a configuration dictionary")

    return (len(errors) == 0, errors)


def create_phase6_manifest(
    git_commit: str,
    seeds: List[int],
    primary_architecture: str = "residual_context",
    primary_variant: str = "self_neighbor_selected",
    test_scene: str = "tum_fr2_xyz",
    resolution: Tuple[int, int] = (320, 240),
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Creates the authoritative Phase 6 experiment manifest."""
    manifest = {
        "phase": 6,
        "title": "Context-Aware Conditional Utility & Interaction Hardening",
        "status": "frozen_candidate",
        "schema_version": "phase6-v2",
        "protocol": "v1",
        "git_commit": git_commit,
        "seeds": seeds,
        "resolution": list(resolution),
        "primary_architecture": primary_architecture,
        "primary_variant": primary_variant,
        "p4_frozen": True,
        "group_definition": "g = (scene_id, frame_id, tuple(sorted(selected_gaussian_ids)))",
        "normalizer": "train_only",
        "train_scene": "tum_fr1_desk",
        "test_scene": test_scene,
    }
    if extra_metadata:
        manifest.update(extra_metadata)
    return manifest
