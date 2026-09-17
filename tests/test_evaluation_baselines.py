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
