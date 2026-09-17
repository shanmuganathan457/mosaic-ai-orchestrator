"""Tests for Benchmark Dataset Loader & Schema Validation."""

from pathlib import Path
import pytest

from mosaic.evaluation.dataset import load_benchmark_dataset
from mosaic.evaluation.models import BenchmarkCase


import tempfile

DATASET_PATH = Path("tests/fixtures/research_dataset/v1/cases.json")


def test_load_benchmark_dataset_success():
    """Verify that cases.json loads successfully and validates against BenchmarkCase schema."""
    cases = load_benchmark_dataset(DATASET_PATH)
    assert len(cases) == 23
    assert all(isinstance(c, BenchmarkCase) for c in cases)
    assert cases[0].case_id == "case_001"
    assert cases[0].expected_final_verdict.value == "ALLOW"


def test_load_benchmark_dataset_file_not_found():
    """Verify ValueError is raised if dataset file does not exist."""
    with pytest.raises(ValueError, match="not found"):
        load_benchmark_dataset("tests/fixtures/non_existent.json")


def test_load_benchmark_dataset_malformed_json():
    """Verify ValueError is raised for malformed non-JSON file."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        tmp.write("INVALID JSON {{{")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValueError, match="Failed to parse dataset JSON"):
            load_benchmark_dataset(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_benchmark_dataset_invalid_schema():
    """Verify ValueError is raised for JSON array with invalid case schema."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        tmp.write('[{"case_id": "case_999"}]')
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValueError, match="failed schema validation"):
            load_benchmark_dataset(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
