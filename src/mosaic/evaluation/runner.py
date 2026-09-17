"""Evaluation Harness Runner Executing Benchmark Datasets across Systems."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from mosaic.evaluation.baselines import (
    BaselineASingleIntentSystem,
    BaselineBDirectMultiAgentSystem,
    BaseResearchSystem,
    MosaicResearchSystem,
)
from mosaic.evaluation.dataset import load_benchmark_dataset
from mosaic.evaluation.metrics import compute_aggregate_metrics
from mosaic.evaluation.models import (
    AggregateMetrics,
    BenchmarkCase,
    EvaluationReport,
    EvaluationSystemResult,
)


class EvaluationRunner:
    """Orchestrates benchmark dataset execution across Baseline A, Baseline B, and MOSAIC systems."""

    def __init__(
        self,
        systems: Optional[List[BaseResearchSystem]] = None,
        dataset_version: str = "v1",
    ) -> None:
        self.systems = systems or [
            BaselineASingleIntentSystem(),
            BaselineBDirectMultiAgentSystem(),
            MosaicResearchSystem(),
        ]
        self.dataset_version = dataset_version

    def run_benchmark(self, dataset_or_path: Union[str, Path, List[BenchmarkCase]]) -> EvaluationReport:
        """Runs all registered systems across benchmark dataset cases and generates structured report."""
        if isinstance(dataset_or_path, (str, Path)):
            cases = load_benchmark_dataset(dataset_or_path)
        else:
            cases = dataset_or_path

        all_results: List[EvaluationSystemResult] = []
        systems_aggregate: Dict[str, AggregateMetrics] = {}

        # Evaluate per system
        for system in self.systems:
            system_results: List[EvaluationSystemResult] = []
            for case in cases:
                res = system.evaluate_case(case)
                system_results.append(res)
                all_results.append(res)

            aggregate = compute_aggregate_metrics(system_results, cases)
            systems_aggregate[system.system_name] = aggregate

        return EvaluationReport(
            dataset_version=self.dataset_version,
            total_cases=len(cases),
            systems=systems_aggregate,
            per_case_results=all_results,
        )

    def run_and_save_report(
        self, dataset_or_path: Union[str, Path, List[BenchmarkCase]], output_path: Union[str, Path]
    ) -> EvaluationReport:
        """Runs evaluation and serializes the resulting report to JSON."""
        report = self.run_benchmark(dataset_or_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        return report
