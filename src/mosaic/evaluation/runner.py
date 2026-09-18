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
    ) -> EvaluationReport:
        """Runs the benchmark using Design C: shared intent decomposition and action compilation.

        For each case:
        1. A single SharedEvaluationContext is built via shared_context_builder.build().
           This makes exactly ONE LLM decomposition call and ONE compilation call per proposal.
        2. All registered systems receive the same SharedEvaluationContext and evaluate independently.

        Research Integrity:
        - Ground truth is never passed to the builder or shared context.
        - Each system applies its own unique downstream logic on the shared artifacts.
        - MOSAIC still runs its full deterministic Validation Engine.
        - Baseline A still takes only the first intent/action from the shared results.
        """
        if isinstance(dataset_or_path, (str, Path)):
            cases = load_benchmark_dataset(dataset_or_path)
        else:
            cases = dataset_or_path

        all_results: List[EvaluationSystemResult] = []
        systems_aggregate: Dict[str, AggregateMetrics] = {}

        # Pre-collect per-system results lists
        system_results_map: Dict[str, List[EvaluationSystemResult]] = {
            system.system_name: [] for system in self.systems
        }

        for case in cases:
            # ONE shared context build per case (1 decomposition + N compilations)
            shared = shared_context_builder.build(
                customer_message=case.customer_message,
                initial_facts=dict(case.initial_facts),
                active_flags=list(case.active_flags),
            )

            # All systems evaluate using the shared context
            for system in self.systems:
                res = system.evaluate_case_with_shared_context(case, shared)
                system_results_map[system.system_name].append(res)
                all_results.append(res)

        # Compute aggregate metrics per system
        for system in self.systems:
            results = system_results_map[system.system_name]
            aggregate = compute_aggregate_metrics(results, cases)
            systems_aggregate[system.system_name] = aggregate

        return EvaluationReport(
            dataset_version=self.dataset_version,
            provider_name=self.provider_name,
            model_name=self.model_name,
            total_cases=len(cases),
            systems=systems_aggregate,
            per_case_results=all_results,
        )

    def run_and_save_report_shared(
        self,
        dataset_or_path: Union[str, Path, List[BenchmarkCase]],
        shared_context_builder: SharedContextBuilder,
        output_path: Union[str, Path],
    ) -> EvaluationReport:
        """Runs shared-context evaluation and serializes the resulting report to JSON."""
        report = self.run_benchmark_shared(dataset_or_path, shared_context_builder)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        return report

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
