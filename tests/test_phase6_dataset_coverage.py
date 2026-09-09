"""Tests for Phase 6 Context-Centric Dataset Coverage and Invariants.

Verifies:
  1. Candidate pool invariance: P_t = candidate_pool_ids
  2. Candidate pool consistency: all records in same exact context share identical candidate_pool_ids
  3. 100% exact candidate coverage: M_t == P_t (zero missing)
  4. Zero duplicate measurements: |unique(M_t)| == |M_t|
  5. Exact context identity: S_1 == S_2 within the same group key
  6. Context disjointness: S_t ∩ P_t = ∅
  7. Cost validity: ΔT > 0 or handled via effective positive baseline cost (no silent abs)
  8. Empty context identity: S=∅ ⟹ U*(i|∅) == U*_single(i) and r*_i(∅) == 0
  9. Split-integrity: No exact group split across train/val/test
  10. Rank stability evaluator: Zero synthetic baseline filling
"""
import os
import json
import pytest
import numpy as np
from typing import Dict, List, Tuple, Any

from research.phase6_dataset import prepare_phase6_splits


@pytest.fixture(scope="module")
def dataset_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.environ.get("PHASE6_TEST_DATASET")
    if env_path and os.path.exists(env_path):
        return env_path
    path = os.path.join(
        repo_root, "results", "phase6_context_utility", "datasets", "conditional_oracle_seed_42.json"
    )
    if not os.path.exists(path):
        pilot_path = os.path.join(
            repo_root, "results", "phase6_context_utility", "datasets", "pilot_verification.json"
        )
        if os.path.exists(pilot_path):
            return pilot_path
        pytest.skip(f"Dataset not found at: {path}")
    return path


@pytest.fixture(scope="module")
def dataset_samples(dataset_path):
    with open(dataset_path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def exact_context_groups(dataset_samples):
    groups: Dict[Tuple[str, int, Tuple[int, ...]], List[Dict[str, Any]]] = {}
    for s in dataset_samples:
        ctx_ids = tuple(sorted(int(x) for x in (s.get("context_ids", []) or [])))
        key = (str(s["scene"]), int(s["frame"]), ctx_ids)
        groups.setdefault(key, []).append(s)
    return groups


def test_candidate_pool_is_fixed(exact_context_groups):
    """Test 1: Candidate pool must be defined and non-empty for every group."""
    assert len(exact_context_groups) > 0
    for key, s_list in exact_context_groups.items():
        pool = s_list[0].get("candidate_pool_ids")
        assert pool is not None, f"Group {key} missing candidate_pool_ids"
        assert len(pool) >= 2, f"Group {key} candidate pool must have at least 2 candidates"


def test_candidate_pool_consistent_within_group(exact_context_groups):
    """Test 2: All records in the same exact context group must share identical candidate_pool_ids."""
    for key, s_list in exact_context_groups.items():
        expected_pool = tuple(s_list[0].get("candidate_pool_ids", []))
        for idx, s in enumerate(s_list):
            cand_pool = tuple(s.get("candidate_pool_ids", []))
            assert cand_pool == expected_pool, (
                f"Candidate pool mismatch in group {key} at sample index {idx}: {cand_pool} vs {expected_pool}"
            )


def test_measured_ids_equals_candidate_pool(exact_context_groups):
    """Test 3: Measured IDs must exactly equal candidate pool (M_t == P_t, zero missing)."""
    for key, s_list in exact_context_groups.items():
        pool_ids = set(s_list[0].get("candidate_pool_ids", []))
        measured_ids = set(int(s["candidate_id"]) for s in s_list)
        missing = pool_ids - measured_ids
        assert len(missing) == 0, f"Group {key} has missing candidates: {missing}"
        assert measured_ids == pool_ids, f"Group {key} measured {measured_ids} != pool {pool_ids}"


def test_no_duplicate_candidate_measurements(exact_context_groups):
    """Test 4: Zero duplicate candidates within a context group: |unique(M_t)| == |M_t|."""
    for key, s_list in exact_context_groups.items():
        measured_ids = [int(s["candidate_id"]) for s in s_list]
        unique_ids = set(measured_ids)
        assert len(measured_ids) == len(unique_ids), (
            f"Group {key} has duplicate candidate measurements: {len(measured_ids)} total vs {len(unique_ids)} unique"
        )


def test_exact_context_identity(exact_context_groups):
    """Test 5: Context identity S_t must be identical for all candidates in the group."""
    for key, s_list in exact_context_groups.items():
        expected_ctx = key[2]
        for s in s_list:
            ctx = tuple(sorted(int(x) for x in (s.get("context_ids", []) or [])))
            assert ctx == expected_ctx, f"Context mismatch in group {key}: {ctx} vs {expected_ctx}"


def test_context_disjoint_from_candidate_pool(dataset_samples):
    """Test 6: S_t ∩ P_t = ∅ across every sample."""
    for s in dataset_samples:
        ctx_ids = set(int(x) for x in (s.get("context_ids", []) or []))
        pool_ids = set(int(x) for x in (s.get("candidate_pool_ids", []) or []))
        cand_id = int(s.get("candidate_id", -1))
        
        # Candidate itself cannot be in context
        assert cand_id not in ctx_ids, f"Candidate {cand_id} found inside its own context {ctx_ids}"
        # Context cannot overlap with candidate pool
        overlap = ctx_ids & pool_ids
        assert len(overlap) == 0, f"Context {ctx_ids} overlaps with candidate pool {pool_ids}: {overlap}"


def test_cost_validity_and_effective_delta_t(dataset_samples):
    """Test 7: Check that effective cost is strictly positive and delta_t_valid is recorded."""
    for s in dataset_samples:
        # Effective cost must be positive and finite
        eff_dt = s.get("effective_delta_t_ms", s.get("delta_t_conditional_ms"))
        assert eff_dt is not None and eff_dt > 0, f"Effective delta_t must be positive, got {eff_dt}"
        assert not np.isnan(eff_dt) and not np.isinf(eff_dt)
        # Utility must be finite
        u = s["utility_conditional"]
        assert not np.isnan(u) and not np.isinf(u)


def test_empty_context_matches_single_measurement(dataset_samples):
    """Test 8: S = ∅ recovers single measurement and residual target r*_i(∅) == 0."""
    empty_samples = [s for s in dataset_samples if s.get("context_size", 0) == 0 or s.get("context_type") == "empty"]
    assert len(empty_samples) > 0, "Dataset must contain empty context samples"
    for s in empty_samples:
        if "utility_single" in s:
            assert abs(s["utility_conditional"] - s["utility_single"]) < 1e-6, (
                f"Empty context utility {s['utility_conditional']} != single utility {s['utility_single']}"
            )
        if "delta_q_single" in s:
            assert abs(s["delta_q_conditional"] - s["delta_q_single"]) < 1e-6, (
                f"Empty context delta_q {s['delta_q_conditional']} != single delta_q {s['delta_q_single']}"
            )


def test_exact_group_split_integrity(dataset_path):
    """Test 9: An exact group cannot be split across train/val/test splits."""
    # prepare_phase6_splits enforces this by raising ValueError if split is inconsistent
    try:
        train_ds, val_ds, test_ds, normalizer = prepare_phase6_splits([dataset_path])
        assert len(train_ds) > 0 or len(test_ds) > 0
    except ValueError as e:
        pytest.fail(f"Split-integrity violation detected: {e}")


def test_no_synthetic_filling_in_rank_stability(dataset_path, tmp_path):
    """Test 10: Rank stability evaluator operates with zero synthetic baseline filling."""
    from experiments.run_phase6_rank_stability import run_rank_stability_analysis
    out_file = os.path.join(tmp_path, "test_rank_stability.json")
    results = run_rank_stability_analysis(dataset_path, out_file)
    audit = results["candidate_coverage_audit"]
    assert audit["synthetic_baseline_fill_used"] is False
    assert audit["total_missing_candidates"] == 0
    assert audit["total_duplicate_candidates"] == 0
    assert audit["invariant_measured_equals_pool"] is True
    assert results["n_full_coverage_groups"] == results["n_total_evaluated_groups"]
