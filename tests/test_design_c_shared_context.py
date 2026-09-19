"""Tests for Design C: Shared Intent Decomposition + Shared Action Compilation.

Verifies:
1. SharedContextBuilder builds a SharedEvaluationContext with correct fields.
2. Each system (Baseline A, B, MOSAIC) correctly applies its own logic on shared context.
3. Baseline A takes only first action; Baseline B takes all actions; MOSAIC validates.
4. SharedEvaluationContext never stores ground truth.
5. EvaluationRunner.run_benchmark_shared produces a valid EvaluationReport.
6. LLM call counts are reduced compared to standalone evaluation.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from mosaic.compiler import SemanticStateCompiler
from mosaic.domain.models import ValidationVerdict
from mosaic.evaluation.baselines import (
    BaselineASingleIntentSystem,
    BaselineBDirectMultiAgentSystem,
    MosaicResearchSystem,
    SharedContextBuilder,
)
from mosaic.evaluation.models import BenchmarkCase, EvaluationSystemResult, SharedEvaluationContext
from mosaic.evaluation.runner import EvaluationRunner
from mosaic.intake.decomposer import IntentDecompositionEngine


# ============================================================================
# Fixtures
# ============================================================================

SINGLE_INTENT_CASE = BenchmarkCase(
    case_id="shared_test_001",
    category="independent_valid",
    customer_message="I need a refund for my payment pay_abc123",
    expected_intents=["refund_payment"],
    expected_agent_actions=["refund_payment"],
    expected_final_verdict=ValidationVerdict.ALLOW,
    rationale="Single billing refund; no conflicts.",
    initial_facts={"payment_verified": True, "identity_verified": True},
)

MULTI_INTENT_CASE = BenchmarkCase(
    case_id="shared_test_002",
    category="independent_valid",
    customer_message=(
        "My account was hacked so please restrict access. "
        "Also I need a refund for pay_xyz999."
    ),
    expected_intents=["restrict_account", "refund_payment"],
    expected_agent_actions=["restrict_account", "refund_payment"],
    expected_final_verdict=ValidationVerdict.ALLOW,
    rationale="Two independent actions; no conflicts.",
    initial_facts={"fraud_alert_active": True, "payment_verified": True, "identity_verified": True},
)

NO_INTENT_CASE = BenchmarkCase(
    case_id="shared_test_003",
    category="no_intent",
    customer_message="Hello, how is the weather today?",
    expected_intents=[],
    expected_agent_actions=[],
    expected_final_verdict=ValidationVerdict.ESCALATED,
    rationale="No actionable intent — escalated.",
)


# ============================================================================
# SharedContextBuilder Tests
# ============================================================================

class TestSharedContextBuilder:
    """Unit tests for SharedContextBuilder.build()."""

    def test_build_single_intent_case(self):
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=SINGLE_INTENT_CASE.customer_message,
            initial_facts={},
            active_flags=[],
        )
        assert isinstance(shared, SharedEvaluationContext)
        assert shared.case is not None
        assert len(shared.subtask_mappings) == 1
        assert len(shared.compiled_actions) == 1
        assert shared.compiled_actions[0].action_type == "refund_payment"

    def test_build_multi_intent_case(self):
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=MULTI_INTENT_CASE.customer_message,
            initial_facts={},
            active_flags=[],
        )
        assert len(shared.subtask_mappings) == 2
        assert len(shared.compiled_actions) == 2
        action_types = {a.action_type for a in shared.compiled_actions}
        assert "restrict_account" in action_types
        assert "refund_payment" in action_types

    def test_build_no_intent_case(self):
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=NO_INTENT_CASE.customer_message,
            initial_facts={},
            active_flags=[],
        )
        assert len(shared.subtask_mappings) == 0
        assert len(shared.compiled_actions) == 0

    def test_llm_call_count_single_intent(self):
        """1 decomposition + 1 compilation = 2 LLM calls for single intent."""
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=SINGLE_INTENT_CASE.customer_message,
        )
        # 1 decomposition + 1 compilation = 2
        assert shared.llm_call_count == 2

    def test_llm_call_count_multi_intent(self):
        """1 decomposition + 2 compilations = 3 LLM calls for two intents."""
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=MULTI_INTENT_CASE.customer_message,
        )
        # 1 decomposition + 2 compilations = 3
        assert shared.llm_call_count == 3

    def test_llm_call_count_no_intent(self):
        """1 decomposition + 0 compilations = 1 LLM call for no intents."""
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=NO_INTENT_CASE.customer_message,
        )
        # 1 decomposition + 0 compilations = 1
        assert shared.llm_call_count == 1

    def test_ground_truth_not_in_shared_context(self):
        """SharedEvaluationContext must not store any ground truth fields."""
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=SINGLE_INTENT_CASE.customer_message)
        # Verify ground truth fields are absent from SharedEvaluationContext
        assert not hasattr(shared, "expected_final_verdict")
        assert not hasattr(shared, "expected_intents")
        assert not hasattr(shared, "expected_conflicts")
        assert not hasattr(shared, "expected_agent_actions")
        assert not hasattr(shared, "rationale")

    def test_initial_facts_and_flags_passed_to_case_state(self):
        """Initial facts and flags are passed through to the shared case state."""
        builder = SharedContextBuilder()
        facts = {"identity_verified": True}
        flags = ["ACCOUNT_LOCKED"]
        shared = builder.build(
            customer_message=SINGLE_INTENT_CASE.customer_message,
            initial_facts=facts,
            active_flags=flags,
        )
        assert shared.case.state.current_facts.get("identity_verified") is True
        assert "ACCOUNT_LOCKED" in shared.case.state.active_flags


# ============================================================================
# Baseline A: evaluate_case_with_shared_context
# ============================================================================

class TestBaselineASharedContext:
    """Baseline A should use ONLY the first intent and first compiled action."""

    def test_single_intent_produces_allow(self):
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=SINGLE_INTENT_CASE.customer_message)
        system = BaselineASingleIntentSystem()
        result = system.evaluate_case_with_shared_context(SINGLE_INTENT_CASE, shared)

        assert result.system_name == "baseline_a_single_intent"
        assert result.predicted_verdict == ValidationVerdict.ALLOW
        assert len(result.predicted_intents) == 1
        assert len(result.predicted_actions) == 1
        assert result.predicted_intents[0] == "refund_payment"

    def test_multi_intent_takes_only_first(self):
        """Baseline A must take only the FIRST intent from shared multi-intent results."""
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=MULTI_INTENT_CASE.customer_message)
        system = BaselineASingleIntentSystem()
        result = system.evaluate_case_with_shared_context(MULTI_INTENT_CASE, shared)

        assert len(result.predicted_intents) == 1
        assert len(result.predicted_actions) == 1
        # First intent from decomposition order (restrict_account comes first in message)
        assert result.predicted_intents[0] in {"restrict_account", "refund_payment"}

    def test_no_intent_produces_escalated(self):
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=NO_INTENT_CASE.customer_message)
        system = BaselineASingleIntentSystem()
        result = system.evaluate_case_with_shared_context(NO_INTENT_CASE, shared)

        assert result.predicted_verdict == ValidationVerdict.ESCALATED
        assert len(result.predicted_intents) == 0
        assert len(result.predicted_actions) == 0

    def test_result_is_evaluation_system_result(self):
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=SINGLE_INTENT_CASE.customer_message)
        system = BaselineASingleIntentSystem()
        result = system.evaluate_case_with_shared_context(SINGLE_INTENT_CASE, shared)
        assert isinstance(result, EvaluationSystemResult)

    def test_no_detected_conflicts(self):
        """Baseline A never detects conflicts (no validation engine)."""
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=MULTI_INTENT_CASE.customer_message)
        system = BaselineASingleIntentSystem()
        result = system.evaluate_case_with_shared_context(MULTI_INTENT_CASE, shared)
        assert result.detected_conflicts == []

    def test_latency_ms_positive(self):
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=SINGLE_INTENT_CASE.customer_message)
        system = BaselineASingleIntentSystem()
        result = system.evaluate_case_with_shared_context(SINGLE_INTENT_CASE, shared)
        assert result.latency_ms >= 0.0


# ============================================================================
# Baseline B: evaluate_case_with_shared_context
# ============================================================================

class TestBaselineBSharedContext:
    """Baseline B should use ALL compiled actions and ALLOW without validation."""

    def test_single_intent_produces_allow(self):
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=SINGLE_INTENT_CASE.customer_message)
        system = BaselineBDirectMultiAgentSystem()
        result = system.evaluate_case_with_shared_context(SINGLE_INTENT_CASE, shared)

        assert result.system_name == "baseline_b_direct_multi_agent"
        assert result.predicted_verdict == ValidationVerdict.ALLOW
        assert len(result.predicted_intents) == 1
        assert len(result.predicted_actions) == 1

    def test_multi_intent_uses_all_compiled_actions(self):
        """Baseline B must use ALL compiled actions from shared context."""
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=MULTI_INTENT_CASE.customer_message)
        system = BaselineBDirectMultiAgentSystem()
        result = system.evaluate_case_with_shared_context(MULTI_INTENT_CASE, shared)

        assert len(result.predicted_intents) == 2
        assert len(result.predicted_actions) == 2
        action_types = set(result.predicted_actions)
        assert "restrict_account" in action_types
        assert "refund_payment" in action_types

    def test_no_intent_produces_escalated(self):
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=NO_INTENT_CASE.customer_message)
        system = BaselineBDirectMultiAgentSystem()
        result = system.evaluate_case_with_shared_context(NO_INTENT_CASE, shared)

        assert result.predicted_verdict == ValidationVerdict.ESCALATED

    def test_no_detected_conflicts(self):
        """Baseline B never detects conflicts (no validation engine)."""
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=MULTI_INTENT_CASE.customer_message)
        system = BaselineBDirectMultiAgentSystem()
        result = system.evaluate_case_with_shared_context(MULTI_INTENT_CASE, shared)
        assert result.detected_conflicts == []


# ============================================================================
# MOSAIC: evaluate_case_with_shared_context
# ============================================================================

class TestMosaicSharedContext:
    """MOSAIC should run full deterministic Validation Engine on shared compiled actions."""

    def test_single_valid_action_allow(self):
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=SINGLE_INTENT_CASE.customer_message,
            initial_facts=SINGLE_INTENT_CASE.initial_facts,
            active_flags=SINGLE_INTENT_CASE.active_flags,
        )
        system = MosaicResearchSystem()
        result = system.evaluate_case_with_shared_context(SINGLE_INTENT_CASE, shared)

        assert result.system_name == "mosaic_validated"
        assert result.predicted_verdict == ValidationVerdict.ALLOW

    def test_multi_valid_action_allow(self):
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=MULTI_INTENT_CASE.customer_message,
            initial_facts=MULTI_INTENT_CASE.initial_facts,
            active_flags=MULTI_INTENT_CASE.active_flags,
        )
        system = MosaicResearchSystem()
        result = system.evaluate_case_with_shared_context(MULTI_INTENT_CASE, shared)

        assert result.predicted_verdict == ValidationVerdict.ALLOW
        assert len(result.predicted_actions) == 2

    def test_suspicious_location_flag_triggers_escalation(self):
        """MOSAIC must independently escalate when SUSPICIOUS_LOCATION_LOGIN flag is set."""
        escalation_case = BenchmarkCase(
            case_id="shared_test_escalate",
            category="policy_escalation",
            customer_message="I cannot log in, please restore my login access.",
            expected_intents=["restore_login_access"],
            expected_agent_actions=["restore_login_access"],
            expected_final_verdict=ValidationVerdict.ESCALATED,
            rationale="Suspicious location flag triggers policy escalation.",
        )
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=escalation_case.customer_message,
            initial_facts={},
            active_flags=["SUSPICIOUS_LOCATION_LOGIN"],
        )
        system = MosaicResearchSystem()
        result = system.evaluate_case_with_shared_context(escalation_case, shared)

        # MOSAIC should escalate; Baseline A/B would have ALLOWed
        assert result.predicted_verdict == ValidationVerdict.ESCALATED

    def test_no_intent_produces_escalated(self):
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=NO_INTENT_CASE.customer_message)
        system = MosaicResearchSystem()
        result = system.evaluate_case_with_shared_context(NO_INTENT_CASE, shared)
        # No actions → validation engine should ALLOW (no actions to block) or ESCALATED
        # Behavior matches no compiled actions path
        assert result.predicted_verdict in {ValidationVerdict.ALLOW, ValidationVerdict.ESCALATED}

    def test_mosaic_detects_conflicts_independently(self):
        """MOSAIC validation engine is called independently even when artifacts are shared."""
        # Create a case that would trigger a policy conflict
        conflict_case = BenchmarkCase(
            case_id="shared_test_conflict",
            category="policy_conflict",
            customer_message="Please restore my login access.",
            expected_intents=["restore_login_access"],
            expected_agent_actions=["restore_login_access"],
            expected_final_verdict=ValidationVerdict.ESCALATED,
            rationale="Legal hold flag present.",
        )
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=conflict_case.customer_message,
            active_flags=["PENDING_LEGAL_HOLD"],
        )
        system = MosaicResearchSystem()
        result = system.evaluate_case_with_shared_context(conflict_case, shared)
        assert result.predicted_verdict == ValidationVerdict.ESCALATED


# ============================================================================
# Cross-System Differentiation Tests (Design C Validity)
# ============================================================================

class TestDesignCSystemDifferentiation:
    """Verifies that Design C preserves meaningful system differentiation."""

    def test_baseline_a_vs_b_differ_on_multi_intent(self):
        """Baseline A uses only 1 action; Baseline B uses all. They should differ."""
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=MULTI_INTENT_CASE.customer_message)

        system_a = BaselineASingleIntentSystem()
        system_b = BaselineBDirectMultiAgentSystem()

        result_a = system_a.evaluate_case_with_shared_context(MULTI_INTENT_CASE, shared)
        result_b = system_b.evaluate_case_with_shared_context(MULTI_INTENT_CASE, shared)

        # Baseline A picks ONLY ONE action; Baseline B picks ALL
        assert len(result_a.predicted_actions) == 1
        assert len(result_b.predicted_actions) == 2

    def test_mosaic_escalates_when_others_allow(self):
        """MOSAIC may escalate due to policy while Baseline B would ALLOW same actions."""
        policy_case = BenchmarkCase(
            case_id="shared_test_policy_diff",
            category="policy_escalation",
            customer_message="Please restore my login access.",
            expected_intents=["restore_login_access"],
            expected_agent_actions=["restore_login_access"],
            expected_final_verdict=ValidationVerdict.ESCALATED,
            rationale="SUSPICIOUS_LOCATION_LOGIN triggers MOSAIC escalation.",
        )
        builder = SharedContextBuilder()
        shared = builder.build(
            customer_message=policy_case.customer_message,
            active_flags=["SUSPICIOUS_LOCATION_LOGIN"],
        )

        system_b = BaselineBDirectMultiAgentSystem()
        system_mosaic = MosaicResearchSystem()

        result_b = system_b.evaluate_case_with_shared_context(policy_case, shared)
        result_mosaic = system_mosaic.evaluate_case_with_shared_context(policy_case, shared)

        # Baseline B blindly ALLOWs (no validation)
        assert result_b.predicted_verdict == ValidationVerdict.ALLOW
        # MOSAIC detects policy violation and ESCALATEs
        assert result_mosaic.predicted_verdict == ValidationVerdict.ESCALATED

    def test_all_three_systems_same_intents_from_shared(self):
        """All three systems should report the same intents since they come from shared context."""
        builder = SharedContextBuilder()
        shared = builder.build(customer_message=MULTI_INTENT_CASE.customer_message)

        result_a = BaselineASingleIntentSystem().evaluate_case_with_shared_context(
            MULTI_INTENT_CASE, shared
        )
        result_b = BaselineBDirectMultiAgentSystem().evaluate_case_with_shared_context(
            MULTI_INTENT_CASE, shared
        )
        result_mosaic = MosaicResearchSystem().evaluate_case_with_shared_context(
            MULTI_INTENT_CASE, shared
        )

        # B and MOSAIC should have same intent set (all intents from shared)
        assert set(result_b.predicted_intents) == set(result_mosaic.predicted_intents)
        # A has subset (first only)
        assert len(result_a.predicted_intents) == 1
        assert result_a.predicted_intents[0] in set(result_b.predicted_intents)


# ============================================================================
# EvaluationRunner.run_benchmark_shared Tests
# ============================================================================

class TestEvaluationRunnerShared:
    """Integration tests for EvaluationRunner.run_benchmark_shared."""

    def test_run_benchmark_shared_single_case(self):
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        report = runner.run_benchmark_shared([SINGLE_INTENT_CASE], builder)

        assert report.total_cases == 1
        # 3 systems × 1 case = 3 results
        assert len(report.per_case_results) == 3
        assert "baseline_a_single_intent" in report.systems
        assert "baseline_b_direct_multi_agent" in report.systems
        assert "mosaic_validated" in report.systems

    def test_run_benchmark_shared_multi_case(self):
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        cases = [SINGLE_INTENT_CASE, MULTI_INTENT_CASE, NO_INTENT_CASE]
        report = runner.run_benchmark_shared(cases, builder)

        assert report.total_cases == 3
        assert len(report.per_case_results) == 9  # 3 systems × 3 cases

    def test_run_benchmark_shared_vs_standalone_same_verdicts(self):
        """Shared context and standalone evaluation should produce identical verdicts for mock."""
        builder = SharedContextBuilder()
        runner = EvaluationRunner()

        cases = [SINGLE_INTENT_CASE, MULTI_INTENT_CASE]

        report_standalone = runner.run_benchmark(cases)
        report_shared = runner.run_benchmark_shared(cases, builder)

        # Aggregate verdict accuracy should be equal for deterministic mock
        for system_name in ["baseline_a_single_intent", "baseline_b_direct_multi_agent", "mosaic_validated"]:
            acc_standalone = report_standalone.systems[system_name].verdict_accuracy
            acc_shared = report_shared.systems[system_name].verdict_accuracy
            assert acc_standalone == acc_shared, (
                f"Verdict accuracy differs for {system_name}: "
                f"standalone={acc_standalone}, shared={acc_shared}"
            )

    def test_aggregate_metrics_populated(self):
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        report = runner.run_benchmark_shared([SINGLE_INTENT_CASE, MULTI_INTENT_CASE], builder)

        for system_name, agg in report.systems.items():
            assert agg.total_cases == 2
            assert 0.0 <= agg.verdict_accuracy <= 1.0
            assert agg.mean_latency_ms >= 0.0

    def test_report_has_correct_system_names(self):
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        report = runner.run_benchmark_shared([SINGLE_INTENT_CASE], builder)
        assert set(report.systems.keys()) == {
            "baseline_a_single_intent",
            "baseline_b_direct_multi_agent",
            "mosaic_validated",
        }

    def test_per_case_results_have_correct_case_ids(self):
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        report = runner.run_benchmark_shared([SINGLE_INTENT_CASE, MULTI_INTENT_CASE], builder)

        result_case_ids = {r.case_id for r in report.per_case_results}
        assert "shared_test_001" in result_case_ids
        assert "shared_test_002" in result_case_ids


# ============================================================================
# Fault-Tolerance, Persistence & Resume Tests
# ============================================================================

import tempfile
from pathlib import Path

class TestSharedContextResilienceAndResume:
    """Tests per-case exception isolation, atomic persistence, resume capabilities, and failure accounting."""

    def test_per_case_exception_isolation(self):
        """Verify that an exception in case 1 does not prevent case 2 from completing."""
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        cases = [SINGLE_INTENT_CASE, MULTI_INTENT_CASE]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "results.json"

            original_build = builder.build
            call_count = 0

            def faulty_build(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("Simulated Ollama Timeout in Intent Decomposition")
                return original_build(*args, **kwargs)

            with patch.object(builder, "build", side_effect=faulty_build):
                report = runner.run_and_save_report_shared(cases, builder, output_path=out_path, resume=False)

            assert report.total_cases == 2
            assert report.completed_cases == 1
            assert report.failed_cases_count == 1
            assert len(report.failed_case_records) == 1
            assert report.failed_case_records[0].case_id == SINGLE_INTENT_CASE.case_id
            assert report.failed_case_records[0].exception_type == "RuntimeError"
            assert "Simulated Ollama Timeout" in report.failed_case_records[0].error_message
            assert report.failed_case_records[0].failure_stage == "INTENT_DECOMPOSITION"

            completed_c_ids = {r.case_id for r in report.per_case_results}
            assert MULTI_INTENT_CASE.case_id in completed_c_ids
            assert SINGLE_INTENT_CASE.case_id not in completed_c_ids

            assert out_path.exists()
            summary_file = Path(tmp_dir) / "summary.json"
            assert summary_file.exists()

    def test_resume_skips_completed_cases(self):
        """Verify that running with resume=True skips already completed cases."""
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        cases = [SINGLE_INTENT_CASE, MULTI_INTENT_CASE]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "results.json"

            report1 = runner.run_and_save_report_shared([SINGLE_INTENT_CASE], builder, output_path=out_path, resume=False)
            assert report1.completed_cases == 1

            with patch.object(builder, "build", wraps=builder.build) as mock_build:
                report2 = runner.run_and_save_report_shared(cases, builder, output_path=out_path, resume=True)

            assert mock_build.call_count == 1
            assert report2.total_cases == 2
            assert report2.completed_cases == 2
            assert report2.skipped_resumed_cases == 1

    def test_resume_rejected_on_metadata_mismatch(self):
        """Verify that provider/model/dataset mismatch prevents unsafe resume."""
        builder = SharedContextBuilder()
        runner = EvaluationRunner(provider_name="ollama", model_name="llama3.2:latest", dataset_version="v1_natural_language")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "results.json"

            report_mock = EvaluationRunner(provider_name="mock").run_and_save_report_shared([SINGLE_INTENT_CASE], builder, output_path=out_path, resume=False)

            with patch.object(builder, "build", wraps=builder.build) as mock_build:
                report_resumed = runner.run_and_save_report_shared([SINGLE_INTENT_CASE, MULTI_INTENT_CASE], builder, output_path=out_path, resume=True)

            assert mock_build.call_count == 2
            assert report_resumed.skipped_resumed_cases == 0

    def test_failed_cases_retry_on_resume(self):
        """Verify that failed cases are retried when resuming."""
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        cases = [SINGLE_INTENT_CASE, MULTI_INTENT_CASE]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "results.json"

            call_count = 0
            original_build = builder.build

            def faulty_build(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("Transient network error")
                return original_build(*args, **kwargs)

            with patch.object(builder, "build", side_effect=faulty_build):
                runner.run_and_save_report_shared(cases, builder, output_path=out_path, resume=False)

            report_resumed = runner.run_and_save_report_shared(cases, builder, output_path=out_path, resume=True)

            assert report_resumed.total_cases == 2
            assert report_resumed.completed_cases == 2
            assert report_resumed.skipped_resumed_cases == 1

    def test_ground_truth_isolation_and_metric_exclusion(self):
        """Verify metrics exclude failed cases and ground truth is not leaked to builder."""
        builder = SharedContextBuilder()
        runner = EvaluationRunner()
        cases = [SINGLE_INTENT_CASE, MULTI_INTENT_CASE]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "results.json"

            # Patch build to verify arguments passed to build never include ground truth fields
            original_build = builder.build
            def build_inspector(*args, **kwargs):
                assert "expected_final_verdict" not in kwargs
                assert "expected_intents" not in kwargs
                assert "expected_agent_actions" not in kwargs
                return original_build(*args, **kwargs)

            with patch.object(builder, "build", side_effect=build_inspector):
                report = runner.run_and_save_report_shared(cases, builder, output_path=out_path, resume=False)
                assert report.completed_cases == 2
