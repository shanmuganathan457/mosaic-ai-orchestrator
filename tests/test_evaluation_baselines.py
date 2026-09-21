"""Tests for Research Baselines (Baseline A, Baseline B, MOSAIC)."""

from pathlib import Path
from mosaic.evaluation.baselines import (
    BaselineASingleIntentSystem,
    BaselineBDirectMultiAgentSystem,
    MosaicResearchSystem,
)
from mosaic.evaluation.dataset import load_benchmark_dataset


DATASET_PATH = Path("tests/fixtures/research_dataset/v1/cases.json")


def test_baseline_a_single_intent_system():
    """Verify Baseline A single-intent routing behavior."""
    cases = load_benchmark_dataset(DATASET_PATH)
    multi_intent_case = next(c for c in cases if c.case_id == "case_001")

    baseline_a = BaselineASingleIntentSystem()
    result = baseline_a.evaluate_case(multi_intent_case)

    assert result.system_name == "baseline_a_single_intent"
    assert len(result.predicted_intents) == 1  # Only 1 intent captured
    assert len(result.predicted_actions) == 1
    assert result.execution_outcome == "DIRECT_SINGLE_ACTION_EXECUTION"


def test_baseline_b_direct_multi_agent_system():
    """Verify Baseline B direct multi-agent synthesis behavior without validation."""
    cases = load_benchmark_dataset(DATASET_PATH)
    conflict_case = next(c for c in cases if c.case_id == "case_003")

    baseline_b = BaselineBDirectMultiAgentSystem()
    result = baseline_b.evaluate_case(conflict_case)

    assert result.system_name == "baseline_b_direct_multi_agent"
    assert len(result.predicted_intents) == 2
    assert len(result.predicted_actions) == 2
    # Baseline B directly ALLOWs both actions without detecting the cross-action postcondition conflict
    assert result.predicted_verdict.value == "ALLOW"
    assert len(result.detected_conflicts) == 0


def test_mosaic_research_system():
    """Verify MOSAIC system with state-aware validation engine."""
    cases = load_benchmark_dataset(DATASET_PATH)
    conflict_case = next(c for c in cases if c.case_id == "case_003")

    mosaic_sys = MosaicResearchSystem()
    result = mosaic_sys.evaluate_case(conflict_case)

    assert result.system_name == "mosaic_validated"
    assert len(result.predicted_intents) == 2
    assert result.predicted_verdict.value == "BLOCK"
    assert len(result.detected_conflicts) > 0


def test_mosaic_evaluate_case_forwards_primary_conflicts():
    """Regression: primary_conflicts must be forwarded from ValidationResult into EvaluationSystemResult.

    Phase 7E audit found that baselines.py omitted primary_conflicts and secondary_conflicts
    from the EvaluationSystemResult constructor. This test proves the plumbing fix is correct.

    Uses case_004 (failed_dependency): restore_login_access with identity_verified=False and
    no verify_identity completed. Expects MISSING_DEPENDENCY as primary conflict.
    """
    cases = load_benchmark_dataset(DATASET_PATH)
    dep_case = next(c for c in cases if c.case_id == "case_004")

    mosaic_sys = MosaicResearchSystem()
    result = mosaic_sys.evaluate_case(dep_case)

    assert result.system_name == "mosaic_validated"
    assert result.predicted_verdict.value == "BLOCK"

    # Primary and secondary must be lists (not None), properly forwarded
    assert isinstance(result.primary_conflicts, list), (
        "primary_conflicts must be a list in EvaluationSystemResult"
    )
    assert isinstance(result.secondary_conflicts, list), (
        "secondary_conflicts must be a list in EvaluationSystemResult"
    )

    # For a BLOCK case with a dependency failure, primary_conflicts must be non-empty
    primary_types = {c.value for c in result.primary_conflicts}
    all_detected_types = {c.value for c in result.detected_conflicts}

    assert len(result.primary_conflicts) > 0, (
        "primary_conflicts must be non-empty when the validation engine detects conflicts: "
        f"detected_conflicts={all_detected_types}"
    )

    # The primary conflict must be a subset of all detected conflicts (no fabricated entries)
    assert primary_types.issubset(all_detected_types), (
        f"primary_conflicts {primary_types} must be a subset of detected_conflicts {all_detected_types}"
    )


def test_mosaic_evaluate_case_primary_secondary_partition_is_exhaustive():
    """Regression: primary + secondary conflicts must together equal all detected conflicts.

    Phase 7E hierarchical semantics guarantee that every detected conflict is classified
    into exactly one of: primary_conflicts or secondary_conflicts.
    """
    cases = load_benchmark_dataset(DATASET_PATH)

    mosaic_sys = MosaicResearchSystem()

    # Test against cases that are expected to produce conflicts
    conflict_cases = [c for c in cases if c.expected_conflicts]
    assert conflict_cases, "Expected at least some cases with conflicts in v1 dataset"

    for case in conflict_cases:
        result = mosaic_sys.evaluate_case(case)

        if not result.detected_conflicts:
            # No conflicts detected — primary and secondary must both be empty
            assert result.primary_conflicts == [], (
                f"[{case.case_id}] primary_conflicts must be empty when no conflicts detected"
            )
            assert result.secondary_conflicts == [], (
                f"[{case.case_id}] secondary_conflicts must be empty when no conflicts detected"
            )
        else:
            all_detected = set(c.value for c in result.detected_conflicts)
            all_primary = set(c.value for c in result.primary_conflicts)
            all_secondary = set(c.value for c in result.secondary_conflicts)

            # primary ∪ secondary must cover all detected (every conflict is classified)
            union = all_primary | all_secondary
            assert union == all_detected, (
                f"[{case.case_id}] primary ∪ secondary ({union}) must equal detected ({all_detected}). "
                f"primary={all_primary}, secondary={all_secondary}"
            )

            # primary ∩ secondary must be empty (no conflict in both)
            intersection = all_primary & all_secondary
            assert not intersection, (
                f"[{case.case_id}] primary ∩ secondary must be empty, got {intersection}"
            )


def test_mosaic_allow_case_has_empty_primary_secondary_conflicts():
    """Regression: for ALLOW cases, primary_conflicts and secondary_conflicts must be empty."""
    cases = load_benchmark_dataset(DATASET_PATH)
    allow_case = next(c for c in cases if c.case_id == "case_001")

    mosaic_sys = MosaicResearchSystem()
    result = mosaic_sys.evaluate_case(allow_case)

    assert result.predicted_verdict.value == "ALLOW"
    assert result.primary_conflicts == [], "ALLOW case must have empty primary_conflicts"
    assert result.secondary_conflicts == [], "ALLOW case must have empty secondary_conflicts"
    assert result.detected_conflicts == [], "ALLOW case must have empty detected_conflicts"
