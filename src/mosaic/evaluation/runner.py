"""Evaluation Harness Runner Executing Benchmark Datasets across Systems."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from mosaic.config.settings import settings
from mosaic.evaluation.baselines import (
    BaselineASingleIntentSystem,
    BaselineBDirectMultiAgentSystem,
    BaseResearchSystem,
    MosaicResearchSystem,
    SharedContextBuilder,
)
from mosaic.evaluation.dataset import load_benchmark_dataset
from mosaic.evaluation.metrics import compute_aggregate_metrics
from mosaic.evaluation.models import (
    AggregateMetrics,
    BenchmarkCase,
    EvaluationReport,
    EvaluationSystemResult,
    SharedEvaluationContext,
)
from mosaic.llm.base import BaseLLMProvider
from mosaic.llm.models import LLMRequest, LLMResponse

logger = logging.getLogger("mosaic.evaluation.runner")


class EvaluationRateLimitedProvider(BaseLLMProvider):
    """Evaluation-only wrapper around BaseLLMProvider enforcing a minimum interval between requests."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        min_interval_seconds: float = settings.GEMINI_EVAL_MIN_INTERVAL_SECONDS,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        time_func=time.time,
        sleep_func=time.sleep,
    ) -> None:
        self.provider = provider
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self._time_func = time_func
        self._sleep_func = sleep_func
        self._last_request_time: Optional[float] = None

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def pop_recorded_usage(self) -> Dict[str, Any]:
        if hasattr(self.provider, "pop_recorded_usage"):
            return self.provider.pop_recorded_usage()
        return {}

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Throttles generate requests ensuring min_interval_seconds between calls, retrying transient 503 errors."""
        attempt = 0
        backoff_delay = 2.0

        while True:
            now = self._time_func()
            if self._last_request_time is not None:
                elapsed = now - self._last_request_time
                if elapsed < self.min_interval_seconds:
                    sleep_duration = self.min_interval_seconds - elapsed
                    logger.info(
                        "Evaluation rate limit pacing: sleeping %.2fs (min_interval=%.2fs)",
                        sleep_duration,
                        self.min_interval_seconds,
                    )
                    self._sleep_func(sleep_duration)

            self._last_request_time = self._time_func()

            try:
                return self.provider.generate(request)
            except Exception as e:
                err_msg = str(e)
                # Check for 503 UNAVAILABLE transient server overload
                is_503 = "503" in err_msg or "UNAVAILABLE" in err_msg.upper()
                # Ensure hard stop on 429 quota exhaustion or non-503 client/validation errors
                is_429 = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg.upper()

                if is_503 and not is_429 and attempt < self.max_retries:
                    attempt += 1
                    logger.warning(
                        "Evaluation transient 503 retry attempt %d/%d after backoff %.1fs. Error: %s",
                        attempt,
                        self.max_retries,
                        backoff_delay,
                        err_msg,
                    )
                    self._sleep_func(backoff_delay)
                    backoff_delay *= self.retry_backoff_factor
                    continue
                raise


class EvaluationRunner:
    """Orchestrates benchmark dataset execution across Baseline A, Baseline B, and MOSAIC systems."""

    def __init__(
        self,
        systems: Optional[List[BaseResearchSystem]] = None,
        dataset_version: str = "v1",
        provider_name: str = "mock",
        model_name: str = "mock-deterministic-v1",
    ) -> None:
        self.systems = systems or [
            BaselineASingleIntentSystem(),
            BaselineBDirectMultiAgentSystem(),
            MosaicResearchSystem(),
        ]
        self.dataset_version = dataset_version
        self.provider_name = provider_name
        self.model_name = model_name

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
            provider_name=self.provider_name,
            model_name=self.model_name,
            total_cases=len(cases),
            systems=systems_aggregate,
            per_case_results=all_results,
        )

    def run_benchmark_shared(
        self,
        dataset_or_path: Union[str, Path, List[BenchmarkCase]],
        shared_context_builder: SharedContextBuilder,
        output_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
    ) -> EvaluationReport:
        """Runs the benchmark using Design C: shared intent decomposition and action compilation.

        For each case:
        1. Exception isolation: A failure in one case is caught, recorded, and does not stop subsequent cases.
        2. Persistence: Partial progress is atomically written after every case.
        3. Resume: Resumes cleanly if an existing report matches provider, model, dataset, and evaluation_design='shared_context'.
        """
        if isinstance(dataset_or_path, (str, Path)):
            cases = load_benchmark_dataset(dataset_or_path)
        else:
            cases = dataset_or_path

        out_path = Path(output_path) if output_path else None
        summary_path = out_path.parent / "summary.json" if out_path else None

        existing_report: Optional[EvaluationReport] = None
        if resume and out_path and out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    report_candidate = EvaluationReport.model_validate(data)
                    # Validate design/provider/model/dataset matching for safe resume
                    if (
                        report_candidate.provider_name == self.provider_name
                        and report_candidate.model_name == self.model_name
                        and report_candidate.dataset_version == self.dataset_version
                        and getattr(report_candidate, "evaluation_design", "") == "shared_context"
                    ):
                        existing_report = report_candidate
                        logger.info("Resuming shared-context benchmark from %s", out_path)
                    else:
                        logger.warning("Existing report metadata mismatch. Starting fresh run.")
            except Exception as e:
                logger.warning("Failed to load existing report for resume: %s. Starting fresh run.", e)

        # Initialize tracking from existing report or fresh state
        completed_case_ids: Set[str] = set()
        system_results_map: Dict[str, List[EvaluationSystemResult]] = {
            system.system_name: [] for system in self.systems
        }
        failed_records: List[Any] = []
        successful_llm_calls = 0
        failed_llm_calls = 0
        skipped_resumed_cases = 0

        if existing_report:
            failed_records = list(existing_report.failed_case_records)
            successful_llm_calls = existing_report.successful_llm_calls
            failed_llm_calls = existing_report.failed_llm_calls

            # Map existing per-case results per system
            existing_case_ids_per_system: Dict[str, Set[str]] = {s.system_name: set() for s in self.systems}
            for res in existing_report.per_case_results:
                if res.system_name in system_results_map:
                    system_results_map[res.system_name].append(res)
                    existing_case_ids_per_system[res.system_name].add(res.case_id)

            # Find cases where ALL registered systems have completed
            all_sys_names = {s.system_name for s in self.systems}
            for c_id in {res.case_id for res in existing_report.per_case_results}:
                if all(c_id in existing_case_ids_per_system[sys_name] for sys_name in all_sys_names):
                    completed_case_ids.add(c_id)

            skipped_resumed_cases = len(completed_case_ids)

        def _save_state():
            """Helper to save results atomically after every case."""
            if not out_path:
                return

            # Re-compute aggregate metrics per system for current completed results
            systems_agg: Dict[str, AggregateMetrics] = {}
            for sys_obj in self.systems:
                res_list = system_results_map[sys_obj.system_name]
                evaluated_c_ids = {r.case_id for r in res_list}
                matching_cases = [c for c in cases if c.case_id in evaluated_c_ids]
                systems_agg[sys_obj.system_name] = compute_aggregate_metrics(res_list, matching_cases)

            all_per_case_results = []
            for sys_obj in self.systems:
                all_per_case_results.extend(system_results_map[sys_obj.system_name])

            rep = EvaluationReport(
                dataset_version=self.dataset_version,
                provider_name=self.provider_name,
                model_name=self.model_name,
                evaluation_design="shared_context",
                total_cases=len(cases),
                completed_cases=len(completed_case_ids),
                failed_cases_count=len(failed_records),
                skipped_resumed_cases=skipped_resumed_cases,
                successful_llm_calls=successful_llm_calls,
                failed_llm_calls=failed_llm_calls,
                systems=systems_agg,
                per_case_results=all_per_case_results,
                failed_case_records=failed_records,
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_results = out_path.with_suffix(".tmp")
            with open(tmp_results, "w", encoding="utf-8") as f:
                f.write(rep.model_dump_json(indent=2))
            tmp_results.replace(out_path)

            if summary_path:
                tmp_summary = summary_path.with_suffix(".tmp")
                with open(tmp_summary, "w", encoding="utf-8") as f:
                    json.dump({
                        "provider_name": self.provider_name,
                        "model_name": self.model_name,
                        "dataset_version": self.dataset_version,
                        "evaluation_design": "shared_context",
                        "total_cases": rep.total_cases,
                        "completed_cases": rep.completed_cases,
                        "failed_cases_count": rep.failed_cases_count,
                        "skipped_resumed_cases": rep.skipped_resumed_cases,
                        "successful_llm_calls": rep.successful_llm_calls,
                        "failed_llm_calls": rep.failed_llm_calls,
                        "systems": {k: v.model_dump() for k, v in rep.systems.items()},
                    }, f, indent=2)
                tmp_summary.replace(summary_path)

        for case in cases:
            if case.case_id in completed_case_ids:
                logger.info("Skipping already completed case_id=%s (resume mode)", case.case_id)
                continue

            try:
                # ONE shared context build per case (1 decomposition + N compilations)
                shared = shared_context_builder.build(
                    customer_message=case.customer_message,
                    initial_facts=dict(case.initial_facts),
                    active_flags=list(case.active_flags),
                )
                successful_llm_calls += shared.llm_call_count

                # All systems evaluate using the shared context
                for system in self.systems:
                    res = system.evaluate_case_with_shared_context(case, shared)
                    system_results_map[system.system_name].append(res)

                completed_case_ids.add(case.case_id)
                _save_state()

            except Exception as e:
                failed_llm_calls += 1
                logger.error("Error evaluating benchmark case_id=%s: %s", case.case_id, e)

                # Determine failure stage
                err_str = str(e)
                failure_stage = "INTENT_DECOMPOSITION" if "decomposition" in err_str.lower() else ("ACTION_COMPILATION" if "compilation" in err_str.lower() else "SYSTEM_EVALUATION")

                from mosaic.evaluation.models import FailedCaseRecord
                failed_records.append(FailedCaseRecord(
                    case_id=case.case_id,
                    failure_stage=failure_stage,
                    exception_type=type(e).__name__,
                    error_message=str(e)
                ))
                _save_state()
                continue

        # Final build of report
        systems_aggregate: Dict[str, AggregateMetrics] = {}
        for system in self.systems:
            results = system_results_map[system.system_name]
            evaluated_case_ids = {r.case_id for r in results}
            matching_cases = [c for c in cases if c.case_id in evaluated_case_ids]
            aggregate = compute_aggregate_metrics(results, matching_cases)
            systems_aggregate[system.system_name] = aggregate

        all_results: List[EvaluationSystemResult] = []
        for system in self.systems:
            all_results.extend(system_results_map[system.system_name])

        final_report = EvaluationReport(
            dataset_version=self.dataset_version,
            provider_name=self.provider_name,
            model_name=self.model_name,
            evaluation_design="shared_context",
            total_cases=len(cases),
            completed_cases=len(completed_case_ids),
            failed_cases_count=len(failed_records),
            skipped_resumed_cases=skipped_resumed_cases,
            successful_llm_calls=successful_llm_calls,
            failed_llm_calls=failed_llm_calls,
            systems=systems_aggregate,
            per_case_results=all_results,
            failed_case_records=failed_records,
        )

        _save_state()
        return final_report

    def run_and_save_report_shared(
        self,
        dataset_or_path: Union[str, Path, List[BenchmarkCase]],
        shared_context_builder: SharedContextBuilder,
        output_path: Union[str, Path],
        resume: bool = True,
    ) -> EvaluationReport:
        """Runs shared-context evaluation and serializes the resulting report to JSON."""
        return self.run_benchmark_shared(dataset_or_path, shared_context_builder, output_path=output_path, resume=resume)

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


def main() -> None:
    """CLI Entry Point for MOSAIC Benchmark Evaluation execution."""
    import argparse
    import sys
    import urllib.request
    from mosaic.compiler.llm_compiler import LLMActionCompiler
    from mosaic.intake.llm_decomposer import LLMIntentDecomposer
    from mosaic.llm.factory import LLMProviderFactory
    from mosaic.orchestrator import MosaicOrchestrator

    parser = argparse.ArgumentParser(description="MOSAIC Research Benchmark Evaluation Runner")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "ollama", "gemini"], help="LLM Provider type")
    parser.add_argument("--model", type=str, default=None, help="Model name identifier")
    parser.add_argument("--dataset", type=str, default="v1_natural_language", choices=["v1", "v1_natural_language"], help="Benchmark dataset version")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    provider_name = args.provider.lower()
    dataset_version = args.dataset

    if dataset_version == "v1_natural_language":
        dataset_path = Path("tests/fixtures/research_dataset/v1_natural_language/cases.json")
    else:
        dataset_path = Path("tests/fixtures/research_dataset/v1/cases.json")

    # Strict availability check for Ollama experiment mode
    if provider_name == "ollama":
        base_url = "http://localhost:11434"
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
                if resp.status != 200:
                    print(f"ERROR: Ollama server at {base_url} returned status {resp.status}.", file=sys.stderr)
                    sys.exit(1)
        except Exception as e:
            print(f"CRITICAL ERROR: Requested --provider ollama but Ollama server is unreachable at {base_url}: {e}", file=sys.stderr)
            print("Research integrity error: Refusing to silently substitute MockLLMProvider when Ollama was explicitly requested.", file=sys.stderr)
            sys.exit(1)

        model_name = args.model or "llama3.2"
        llm_p = LLMProviderFactory.get_provider("ollama", model_name=model_name)
        decomposer = LLMIntentDecomposer(llm_provider=llm_p)
        compiler = LLMActionCompiler(llm_provider=llm_p)
        systems = [
            BaselineASingleIntentSystem(intake_engine=decomposer, compiler=compiler),
            BaselineBDirectMultiAgentSystem(intake_engine=decomposer, compiler=compiler),
            MosaicResearchSystem(orchestrator=MosaicOrchestrator(intake_engine=decomposer, compiler=compiler)),
        ]
    elif provider_name == "gemini":
        model_name = args.model or "gemini-2.5-flash"
        try:
            raw_llm_p = LLMProviderFactory.get_provider("gemini", model_name=model_name)
            llm_p = EvaluationRateLimitedProvider(
                provider=raw_llm_p,
                min_interval_seconds=settings.GEMINI_EVAL_MIN_INTERVAL_SECONDS,
            )
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Gemini provider: {e}", file=sys.stderr)
            sys.exit(1)

        decomposer = LLMIntentDecomposer(llm_provider=llm_p)
        compiler = LLMActionCompiler(llm_provider=llm_p)
        systems = [
            BaselineASingleIntentSystem(intake_engine=decomposer, compiler=compiler),
            BaselineBDirectMultiAgentSystem(intake_engine=decomposer, compiler=compiler),
            MosaicResearchSystem(orchestrator=MosaicOrchestrator(intake_engine=decomposer, compiler=compiler)),
        ]

    else:
        model_name = args.model or "mock-deterministic-v1"
        systems = [
            BaselineASingleIntentSystem(),
            BaselineBDirectMultiAgentSystem(),
            MosaicResearchSystem(),
        ]

    output_dir = Path("evaluation_results/phase_7b") / model_name
    output_file = Path(args.output) if args.output else output_dir / "results.json"

    print(f"==================================================")
    print(f"MOSAIC PHASE 7B BENCHMARK EVALUATION")
    print(f"==================================================")
    print(f"Provider:       {provider_name}")
    print(f"Model:          {model_name}")
    print(f"Dataset:        {dataset_version} ({dataset_path})")
    print(f"Output Path:    {output_file}")
    print(f"==================================================")

    runner = EvaluationRunner(
        systems=systems,
        dataset_version=dataset_version,
        provider_name=provider_name,
        model_name=model_name,
    )
    report = runner.run_and_save_report(dataset_path, output_file)

    # Save summary report
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "provider_name": provider_name,
            "model_name": model_name,
            "dataset_version": dataset_version,
            "total_cases": report.total_cases,
            "systems": {k: v.model_dump() for k, v in report.systems.items()},
        }, f, indent=2)

    print("\nAGGREGATE BENCHMARK RESULTS:")
    print(json.dumps({k: v.model_dump() for k, v in report.systems.items()}, indent=2))
    print(f"\nSaved full report to: {output_file}")
    print(f"Saved summary to:     {summary_file}")


if __name__ == "__main__":
    main()
