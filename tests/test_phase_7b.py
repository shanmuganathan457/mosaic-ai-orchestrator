"""Tests for Phase 7B Natural Language Dataset, Runner CLI, and Error Attribution."""

from pathlib import Path
import pytest

from mosaic.evaluation.dataset import load_benchmark_dataset
from mosaic.evaluation.error_attribution import determine_error_category
from mosaic.domain.models import ValidationVerdict
from mosaic.evaluation.runner import EvaluationRunner


NATURAL_LANGUAGE_DATASET_PATH = Path("tests/fixtures/research_dataset/v1_natural_language/cases.json")


def test_natural_language_dataset_loading():
    """Verify natural language benchmark dataset loads and validates schema."""
    cases = load_benchmark_dataset(NATURAL_LANGUAGE_DATASET_PATH)
    assert len(cases) == 39

    # Check source_case_id references
    for case in cases:
        assert case.source_case_id is not None
        assert case.source_case_id.startswith("case_")


def test_error_attribution_taxonomy():
    """Verify error category determination logic across taxonomy categories."""
    # 1. PERFECT MATCH
    cat = determine_error_category(
        predicted_intents=["refund_payment"],
        expected_intents=["refund_payment"],
        predicted_actions=["refund_payment"],
        expected_actions=["refund_payment"],
        predicted_verdict=ValidationVerdict.ALLOW,
        expected_verdict=ValidationVerdict.ALLOW,
    )
    assert cat == "CORRECT_ALL"

    # 2. INCORRECT INTENT EXTRACTION
    cat = determine_error_category(
        predicted_intents=["general_inquiry"],
        expected_intents=["refund_payment"],
        predicted_actions=["refund_payment"],
        expected_actions=["refund_payment"],
        predicted_verdict=ValidationVerdict.ALLOW,
        expected_verdict=ValidationVerdict.ALLOW,
    )
    assert cat == "INCORRECT_INTENT_EXTRACTION"

    # 3. INCORRECT ACTION COMPILATION
    cat = determine_error_category(
        predicted_intents=["refund_payment"],
        expected_intents=["refund_payment"],
        predicted_actions=["cancel_subscription"],
        expected_actions=["refund_payment"],
        predicted_verdict=ValidationVerdict.ALLOW,
        expected_verdict=ValidationVerdict.ALLOW,
    )
    assert cat == "INCORRECT_ACTION_COMPILATION"

    # 4. INCORRECT VALIDATION RESULT
    cat = determine_error_category(
        predicted_intents=["refund_payment"],
        expected_intents=["refund_payment"],
        predicted_actions=["refund_payment"],
        expected_actions=["refund_payment"],
        predicted_verdict=ValidationVerdict.ALLOW,
        expected_verdict=ValidationVerdict.BLOCK,
    )
    assert cat == "INCORRECT_VALIDATION_RESULT"


def test_evaluation_runner_natural_language_mode():
    """Verify EvaluationRunner executes natural language dataset in deterministic mode."""
    runner = EvaluationRunner(dataset_version="v1_natural_language")
    report = runner.run_benchmark(NATURAL_LANGUAGE_DATASET_PATH)

    assert report.dataset_version == "v1_natural_language"
    assert report.total_cases == 39
    assert "mosaic_validated" in report.systems
    assert report.systems["mosaic_validated"].verdict_accuracy > report.systems["baseline_a_single_intent"].verdict_accuracy
