"""Tests for Evaluation Runner & JSON Report Serialization."""

from pathlib import Path
import json

from mosaic.evaluation.dataset import load_benchmark_dataset
from mosaic.evaluation.runner import EvaluationRunner


import tempfile

DATASET_PATH = Path("tests/fixtures/research_dataset/v1/cases.json")


def test_evaluation_runner_execution():
    """Verify EvaluationRunner executes benchmark cases across systems and serializes JSON report."""
    runner = EvaluationRunner()
    report = runner.run_benchmark(DATASET_PATH)

    assert report.dataset_version == "v1"
    assert report.total_cases == 23
    assert "baseline_a_single_intent" in report.systems
    assert "baseline_b_direct_multi_agent" in report.systems
    assert "mosaic_validated" in report.systems

    # Verify MOSAIC achieves higher verdict accuracy than baselines on the benchmark
    assert report.systems["mosaic_validated"].verdict_accuracy > report.systems["baseline_a_single_intent"].verdict_accuracy
    assert report.systems["mosaic_validated"].verdict_accuracy > report.systems["baseline_b_direct_multi_agent"].verdict_accuracy

    # Test JSON report saving
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        out_file = Path(tmp.name)

    try:
        runner.run_and_save_report(DATASET_PATH, out_file)
        assert out_file.exists()

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_cases"] == 23
        assert "systems" in data
    finally:
        if out_file.exists():
            out_file.unlink()
