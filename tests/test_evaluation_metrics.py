"""Tests for Evaluation Metrics Calculation."""

from mosaic.domain.models import ConflictType, ValidationVerdict
from mosaic.evaluation.metrics import compute_aggregate_metrics, compute_precision_recall_f1
from mosaic.evaluation.models import BenchmarkCase, EvaluationSystemResult


def test_precision_recall_f1_computation():
    """Verify precision, recall, and F1 calculations for set comparisons."""
    # Perfect match
    p, r, f1 = compute_precision_recall_f1({"a", "b"}, {"a", "b"})
    assert (p, r, f1) == (1.0, 1.0, 1.0)

    # Empty sets
    p, r, f1 = compute_precision_recall_f1(set(), set())
    assert (p, r, f1) == (1.0, 1.0, 1.0)

    # Partial match
    p, r, f1 = compute_precision_recall_f1({"a", "b"}, {"a", "c"})
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_aggregate_metrics_computation():
    """Verify aggregate metrics accuracy, precision, recall, and escalation rates."""
    dummy_case = BenchmarkCase(
        case_id="c1",
        category="test",
        customer_message="test",
        expected_intents=["test"],
        expected_agent_actions=["test"],
        expected_final_verdict=ValidationVerdict.ALLOW,
        rationale="test",
    )

    res1 = EvaluationSystemResult(
        case_id="c1",
        system_name="sys",
        predicted_intents=["test"],
        predicted_actions=["test"],
        predicted_verdict=ValidationVerdict.ALLOW,
        expected_verdict=ValidationVerdict.ALLOW,
        verdict_correct=True,
        detected_conflicts=[],
        expected_conflicts=[],
        conflict_precision=1.0,
        conflict_recall=1.0,
        conflict_f1=1.0,
        execution_outcome="ALLOW",
    )

    res2 = EvaluationSystemResult(
        case_id="c2",
        system_name="sys",
        predicted_intents=["test"],
        predicted_actions=["test"],
        predicted_verdict=ValidationVerdict.ALLOW,
        expected_verdict=ValidationVerdict.BLOCK,
        verdict_correct=False,
        detected_conflicts=[],
        expected_conflicts=[ConflictType.PRECONDITION_UNSATISFIED],
        conflict_precision=1.0,
        conflict_recall=0.0,
        conflict_f1=0.0,
        execution_outcome="ALLOW",
    )

    metrics = compute_aggregate_metrics([res1, res2], [dummy_case, dummy_case])
    assert metrics.total_cases == 2
    assert metrics.verdict_accuracy == 0.5
