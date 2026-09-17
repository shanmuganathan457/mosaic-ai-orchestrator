"""Evaluation Metrics Calculation Module."""

from typing import Dict, List, Set, Tuple
from mosaic.domain.models import ConflictType, ValidationVerdict
from mosaic.evaluation.models import AggregateMetrics, BenchmarkCase, EvaluationSystemResult


def compute_precision_recall_f1(predicted: Set[str], expected: Set[str]) -> Tuple[float, float, float]:
    """Computes precision, recall, and F1 score between predicted and expected string sets."""
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted or not expected:
        return 0.0, 0.0, 0.0

    tp = len(predicted.intersection(expected))
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_aggregate_metrics(results: List[EvaluationSystemResult], benchmark_cases: List[BenchmarkCase]) -> AggregateMetrics:
    """Computes overall aggregate metrics for a system across all benchmark cases."""
    total_cases = len(results)
    if total_cases == 0:
        return AggregateMetrics(
            total_cases=0,
            verdict_accuracy=0.0,
            intent_precision=0.0,
            intent_recall=0.0,
            intent_f1=0.0,
            conflict_precision=0.0,
            conflict_recall=0.0,
            conflict_f1=0.0,
            escalation_precision=0.0,
            false_escalation_rate=0.0,
        )

    # Verdict Accuracy
    correct_verdicts = sum(1 for r in results if r.verdict_correct)
    verdict_accuracy = round(correct_verdicts / total_cases, 4)

    # Overall Micro-Aggregated Intent Precision, Recall, F1 across dataset
    case_map = {c.case_id: c for c in benchmark_cases}
    total_predicted_intents: Set[Tuple[str, str]] = set()
    total_expected_intents: Set[Tuple[str, str]] = set()

    for r in results:
        for i_name in r.predicted_intents:
            total_predicted_intents.add((r.case_id, i_name))

        bench_case = case_map.get(r.case_id)
        if bench_case:
            for exp_intent in bench_case.expected_intents:
                total_expected_intents.add((r.case_id, exp_intent))

    intent_prec, intent_rec, intent_f1 = compute_precision_recall_f1(
        {f"{cid}:{i}" for cid, i in total_predicted_intents},
        {f"{cid}:{i}" for cid, i in total_expected_intents},
    )

    # Overall Conflict Precision, Recall, F1
    total_predicted_conflicts: Set[Tuple[str, str]] = set()
    total_expected_conflicts: Set[Tuple[str, str]] = set()

    for r in results:
        for c in r.detected_conflicts:
            total_predicted_conflicts.add((r.case_id, c.value))
        for c in r.expected_conflicts:
            total_expected_conflicts.add((r.case_id, c.value))

    prec, rec, f1 = compute_precision_recall_f1(
        {f"{cid}:{c}" for cid, c in total_predicted_conflicts},
        {f"{cid}:{c}" for cid, c in total_expected_conflicts},
    )

    # Escalation Precision & False Escalation Rate
    # Escalation precision: True Escalations / Total Predicted Escalations
    # False Escalation Rate: False Escalations / Total Non-Escalation Ground Truth Cases
    predicted_escalations = sum(1 for r in results if r.predicted_verdict == ValidationVerdict.ESCALATED)
    true_escalations = sum(
        1 for r in results
        if r.predicted_verdict == ValidationVerdict.ESCALATED and r.expected_verdict == ValidationVerdict.ESCALATED
    )
    false_escalations = sum(
        1 for r in results
        if r.predicted_verdict == ValidationVerdict.ESCALATED and r.expected_verdict != ValidationVerdict.ESCALATED
    )
    ground_truth_non_escalated = sum(1 for r in results if r.expected_verdict != ValidationVerdict.ESCALATED)

    escalation_precision = round(true_escalations / predicted_escalations, 4) if predicted_escalations > 0 else 1.0
    false_escalation_rate = round(false_escalations / ground_truth_non_escalated, 4) if ground_truth_non_escalated > 0 else 0.0

    # Action Coverage & Unexpected Action Rate across dataset
    total_expected_actions_count = 0
    covered_expected_actions_count = 0
    total_predicted_actions_count = 0
    unexpected_predicted_actions_count = 0

    for r in results:
        bench_case = case_map.get(r.case_id)
        if bench_case:
            exp_actions = set(bench_case.expected_agent_actions)
            pred_actions = set(r.predicted_actions)
            total_expected_actions_count += len(exp_actions)
            covered_expected_actions_count += len(pred_actions.intersection(exp_actions))
            total_predicted_actions_count += len(r.predicted_actions)
            unexpected_predicted_actions_count += len(pred_actions - exp_actions)

    action_coverage = round(covered_expected_actions_count / total_expected_actions_count, 4) if total_expected_actions_count > 0 else 1.0
    unexpected_action_rate = round(unexpected_predicted_actions_count / total_predicted_actions_count, 4) if total_predicted_actions_count > 0 else 0.0

    # Error attribution counts
    error_counts: Dict[str, int] = {}
    for r in results:
        cat = r.error_category
        error_counts[cat] = error_counts.get(cat, 0) + 1

    return AggregateMetrics(
        total_cases=total_cases,
        verdict_accuracy=verdict_accuracy,
        intent_precision=intent_prec,
        intent_recall=intent_rec,
        intent_f1=intent_f1,
        action_coverage=action_coverage,
        unexpected_action_rate=unexpected_action_rate,
        conflict_precision=prec,
        conflict_recall=rec,
        conflict_f1=f1,
        escalation_precision=escalation_precision,
        false_escalation_rate=false_escalation_rate,
        error_attribution_counts=error_counts,
    )
