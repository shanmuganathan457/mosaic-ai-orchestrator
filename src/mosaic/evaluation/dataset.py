"""Evaluation Dataset Loader and Validator."""

import json
from pathlib import Path
from typing import List, Union

from mosaic.evaluation.models import BenchmarkCase


def load_benchmark_dataset(dataset_path: Union[str, Path]) -> List[BenchmarkCase]:
    """Loads and validates a research benchmark dataset JSON file.

    Args:
        dataset_path: Path to cases.json file.

    Returns:
        List of strongly typed, validated BenchmarkCase models.

    Raises:
        ValueError: If file does not exist, is invalid JSON, or fails schema validation.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise ValueError(f"Benchmark dataset file not found at path: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse dataset JSON at {path}: {str(e)}") from e

    if not isinstance(raw_data, list):
        raise ValueError(f"Benchmark dataset at {path} must contain a top-level JSON array of cases.")

    cases: List[BenchmarkCase] = []
    for idx, raw_case in enumerate(raw_data):
        try:
            case = BenchmarkCase.model_validate(raw_case)
            cases.append(case)
        except Exception as e:
            raise ValueError(f"Benchmark case at index {idx} failed schema validation: {str(e)}") from e

    return cases
