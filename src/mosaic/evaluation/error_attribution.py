"""Error Attribution Taxonomy Evaluator for Research Benchmark Analysis."""

from typing import List, Set
from mosaic.domain.models import ValidationVerdict
from mosaic.evaluation.models import BenchmarkCase


def determine_error_category(
    predicted_intents: List[str],
    expected_intents: List[str],
    predicted_actions: List[str],
    expected_actions: List[str],
    predicted_verdict: ValidationVerdict,
    expected_verdict: ValidationVerdict,
) -> str:
    """Categorizes the primary root cause of a benchmark result into error attribution taxonomy:
    
    1. CORRECT_ALL: Intent, action compilation, and validation verdict match ground truth.
    2. INCORRECT_INTENT_EXTRACTION: Misidentified or missing customer intents.
    3. INCORRECT_ACTION_COMPILATION: Correct intents, but action compilation missed or mutated target actions.
    4. INCORRECT_VALIDATION_RESULT: Correct intents and actions, but validation engine produced wrong verdict.
    5. INCORRECT_FINAL_INTERPRETATION: Secondary interpretation breakdown.
    """
    verdict_correct = (predicted_verdict == expected_verdict)
    intents_match = (set(predicted_intents) == set(expected_intents))
    actions_match = (set(predicted_actions) == set(expected_actions))

    if verdict_correct and intents_match and actions_match:
        return "CORRECT_ALL"

    if not intents_match:
        return "INCORRECT_INTENT_EXTRACTION"

    if not actions_match:
        return "INCORRECT_ACTION_COMPILATION"

    if not verdict_correct:
        return "INCORRECT_VALIDATION_RESULT"

    return "INCORRECT_FINAL_INTERPRETATION"
