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
    dummy_case1 = BenchmarkCase(
        case_id="c1",
        category="test",
        customer_message="test",
        expected_intents=["test"],
        expected_agent_actions=["test"],
        expected_final_verdict=ValidationVerdict.ALLOW,
        rationale="test",
    )
    dummy_case2 = BenchmarkCase(
        case_id="c2",
        category="test",
        customer_message="test",
        expected_intents=["test"],
        expected_agent_actions=["test"],
        expected_final_verdict=ValidationVerdict.BLOCK,
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
        intent_precision=1.0,
        intent_recall=1.0,
        intent_f1=1.0,
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
        intent_precision=1.0,
        intent_recall=1.0,
        intent_f1=1.0,
        detected_conflicts=[],
        expected_conflicts=[ConflictType.PRECONDITION_UNSATISFIED],
        conflict_precision=1.0,
        conflict_recall=0.0,
        conflict_f1=0.0,
        execution_outcome="ALLOW",
    )

    metrics = compute_aggregate_metrics([res1, res2], [dummy_case1, dummy_case2])
    assert metrics.total_cases == 2
    assert metrics.verdict_accuracy == 0.5
    assert metrics.intent_f1 == 1.0
    assert metrics.mean_latency_ms == 0.0
    assert metrics.p95_latency_ms == 0.0
    assert metrics.total_prompt_tokens is None
    assert metrics.total_completion_tokens is None
    assert metrics.total_tokens is None


def test_aggregate_latency_and_token_usage():
    """Verify mean latency, p95 latency, and token usage aggregation across benchmark cases."""
    dummy_case = BenchmarkCase(
        case_id="c1",
        category="test",
        customer_message="test",
        expected_intents=["test"],
        expected_agent_actions=["test"],
        expected_final_verdict=ValidationVerdict.ALLOW,
        rationale="test",
    )

    results = []
    # Create 20 mock system results with distinct latencies and token counts
    for i in range(1, 21):
        results.append(
            EvaluationSystemResult(
                case_id="c1",
                system_name="sys",
                predicted_intents=["test"],
                predicted_actions=["test"],
                predicted_verdict=ValidationVerdict.ALLOW,
                expected_verdict=ValidationVerdict.ALLOW,
                verdict_correct=True,
                intent_precision=1.0,
                intent_recall=1.0,
                intent_f1=1.0,
                detected_conflicts=[],
                expected_conflicts=[],
                conflict_precision=1.0,
                conflict_recall=1.0,
                conflict_f1=1.0,
                execution_outcome="ALLOW",
                latency_ms=float(i * 10),  # 10, 20, ..., 200 ms
                token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            )
        )

    metrics = compute_aggregate_metrics(results, [dummy_case])
    assert metrics.total_cases == 20
    # Mean latency of 10..200 (step 10) is 105.0
    assert metrics.mean_latency_ms == 105.0
    # p95 latency of 20 elements (indices 0..19): rank 19 * 0.95 = 18.05 -> interpolated between 190 and 200 = 190.5
    assert metrics.p95_latency_ms == 190.5
    # Token totals: 20 * 100 = 2000, 20 * 50 = 1000, 20 * 150 = 3000
    assert metrics.total_prompt_tokens == 2000
    assert metrics.total_completion_tokens == 1000
    assert metrics.total_tokens == 3000

