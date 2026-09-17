import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from mosaic.agents import (
    AccessMockAgent,
    BaseMockAgent,
    BillingMockAgent,
    SecurityMockAgent,
    SubscriptionMockAgent,
)
from mosaic.compiler import BaseActionCompiler, SemanticStateCompiler
from mosaic.domain.models import (
    Action,
    CaseState,
    ConflictType,
    ValidationResult,
    ValidationVerdict,
)
from mosaic.evaluation.models import BenchmarkCase, EvaluationSystemResult
from mosaic.evaluation.metrics import compute_precision_recall_f1
from mosaic.intake.decomposer import BaseIntentDecomposer, IntentDecompositionEngine
from mosaic.llm.base import BaseLLMProvider
from mosaic.orchestrator import MosaicOrchestrator


def _collect_token_usage(*providers: Optional[Any]) -> Dict[str, Any]:
    """Flushes and aggregates token usage metadata across all LLM providers used during a case execution."""
    unique_providers = {p for p in providers if p is not None}
    combined_usage: Dict[str, int] = {}
    has_valid_usage = False

    for provider in unique_providers:
        if hasattr(provider, "pop_recorded_usage"):
            usage = provider.pop_recorded_usage()
            if usage and "total_tokens" in usage:
                has_valid_usage = True
                combined_usage["prompt_tokens"] = combined_usage.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
                combined_usage["completion_tokens"] = combined_usage.get("completion_tokens", 0) + usage.get("completion_tokens", 0)
                combined_usage["total_tokens"] = combined_usage.get("total_tokens", 0) + usage.get("total_tokens", 0)

    return combined_usage if has_valid_usage else {}



class BaseResearchSystem(ABC):
    """Abstract interface for research baseline systems."""

    @property
    @abstractmethod
    def system_name(self) -> str:
        """Name of the system variant."""
        pass

    @abstractmethod
    def evaluate_case(self, case: BenchmarkCase) -> EvaluationSystemResult:
        """Evaluates a single benchmark case and returns structured system result."""
        pass


class BaselineASingleIntentSystem(BaseResearchSystem):
    """Baseline A: Single-Intent Routing System.

    Selects ONLY the first extracted intent, routes to one specialist agent, compiles action,
    and produces ALLOW directly without cross-action state validation.
    """

    def __init__(
        self,
        intake_engine: Optional[BaseIntentDecomposer] = None,
        compiler: Optional[BaseActionCompiler] = None,
        agent_registry: Optional[Dict[str, BaseMockAgent]] = None,
    ) -> None:
        self.intake_engine = intake_engine or IntentDecompositionEngine()
        self.compiler = compiler or SemanticStateCompiler()
        self.agent_registry = agent_registry or {
            "SecurityMockAgent": SecurityMockAgent(),
            "BillingMockAgent": BillingMockAgent(),
            "SubscriptionMockAgent": SubscriptionMockAgent(),
            "AccessMockAgent": AccessMockAgent(),
        }

    @property
    def system_name(self) -> str:
        return "baseline_a_single_intent"

    def evaluate_case(self, case: BenchmarkCase) -> EvaluationSystemResult:
        providers = [
            getattr(self.intake_engine, "llm_provider", None),
            getattr(self.compiler, "llm_provider", None),
        ]
        _collect_token_usage(*providers)
        start_time = time.perf_counter()

        case_obj, subtask_mappings = self.intake_engine.create_case_from_intake(
            customer_id="bench_user",
            raw_message=case.customer_message,
            initial_facts=case.initial_facts,
            active_flags=case.active_flags,
        )

        predicted_intents: List[str] = []
        predicted_actions: List[str] = []

        if subtask_mappings:
            # Single intent selection: Take ONLY first intent mapping
            subtask, span, agent_name = subtask_mappings[0]
            predicted_intents.append(span.intent_name)

            agent = self.agent_registry.get(agent_name)
            if agent:
                proposal = agent.process_subtask(subtask, span)
                action = self.compiler.compile_proposal(proposal)
                predicted_actions.append(action.action_type)

        # Baseline A does NOT run cross-action or state validation -> assumes ALLOW if action created, else ESCALATED if no intent
        predicted_verdict = ValidationVerdict.ALLOW if predicted_actions else ValidationVerdict.ESCALATED
        detected_conflicts: List[ConflictType] = []

        prec, rec, f1 = compute_precision_recall_f1(
            {c.value for c in detected_conflicts},
            {c.value for c in case.expected_conflicts},
        )

        i_prec, i_rec, i_f1 = compute_precision_recall_f1(
            set(predicted_intents),
            set(case.expected_intents),
        )

        # Baseline A action metrics & error attribution
        exp_actions = set(case.expected_agent_actions)
        pred_actions = set(predicted_actions)
        act_cov = len(pred_actions.intersection(exp_actions)) / len(exp_actions) if exp_actions else 1.0
        unexp_act_rate = len(pred_actions - exp_actions) / len(pred_actions) if pred_actions else 0.0

        from mosaic.evaluation.error_attribution import determine_error_category
        err_cat = determine_error_category(
            predicted_intents=predicted_intents,
            expected_intents=case.expected_intents,
            predicted_actions=predicted_actions,
            expected_actions=case.expected_agent_actions,
            predicted_verdict=predicted_verdict,
            expected_verdict=case.expected_final_verdict,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        token_usage = _collect_token_usage(*providers)

        return EvaluationSystemResult(
            case_id=case.case_id,
            source_case_id=case.source_case_id,
            system_name=self.system_name,
            predicted_intents=predicted_intents,
            predicted_actions=predicted_actions,
            predicted_verdict=predicted_verdict,
            expected_verdict=case.expected_final_verdict,
            verdict_correct=(predicted_verdict == case.expected_final_verdict),
            intent_precision=i_prec,
            intent_recall=i_rec,
            intent_f1=i_f1,
            action_coverage=round(act_cov, 4),
            unexpected_action_rate=round(unexp_act_rate, 4),
            detected_conflicts=detected_conflicts,
            expected_conflicts=case.expected_conflicts,
            conflict_precision=prec,
            conflict_recall=rec,
            conflict_f1=f1,
            execution_outcome="DIRECT_SINGLE_ACTION_EXECUTION" if predicted_actions else "NO_INTENT_DETECTED",
            error_category=err_cat,
            latency_ms=round(elapsed_ms, 2),
            token_usage=token_usage,
        )


class BaselineBDirectMultiAgentSystem(BaseResearchSystem):
    """Baseline B: Direct Multi-Agent Synthesis System.

    Decomposes message into multiple intents, invokes all specialist agents, compiles all proposed actions,
    and directly executes/ALLOWs all actions WITHOUT calling the MOSAIC Validation Engine.
    """

    def __init__(
        self,
        intake_engine: Optional[BaseIntentDecomposer] = None,
        compiler: Optional[BaseActionCompiler] = None,
        agent_registry: Optional[Dict[str, BaseMockAgent]] = None,
    ) -> None:
        self.intake_engine = intake_engine or IntentDecompositionEngine()
        self.compiler = compiler or SemanticStateCompiler()
        self.agent_registry = agent_registry or {
            "SecurityMockAgent": SecurityMockAgent(),
            "BillingMockAgent": BillingMockAgent(),
            "SubscriptionMockAgent": SubscriptionMockAgent(),
            "AccessMockAgent": AccessMockAgent(),
        }

    @property
    def system_name(self) -> str:
        return "baseline_b_direct_multi_agent"

    def evaluate_case(self, case: BenchmarkCase) -> EvaluationSystemResult:
        providers = [
            getattr(self.intake_engine, "llm_provider", None),
            getattr(self.compiler, "llm_provider", None),
        ]
        _collect_token_usage(*providers)
        start_time = time.perf_counter()

        case_obj, subtask_mappings = self.intake_engine.create_case_from_intake(
            customer_id="bench_user",
            raw_message=case.customer_message,
            initial_facts=case.initial_facts,
            active_flags=case.active_flags,
        )

        predicted_intents: List[str] = []
        predicted_actions: List[str] = []

        for subtask, span, agent_name in subtask_mappings:
            predicted_intents.append(span.intent_name)
            agent = self.agent_registry.get(agent_name)
            if agent:
                proposal = agent.process_subtask(subtask, span)
                action = self.compiler.compile_proposal(proposal)
                predicted_actions.append(action.action_type)

        # Baseline B does NOT perform validation -> directly ALLOWs all compiled actions
        predicted_verdict = ValidationVerdict.ALLOW if predicted_actions else ValidationVerdict.ESCALATED
        detected_conflicts: List[ConflictType] = []

        prec, rec, f1 = compute_precision_recall_f1(
            {c.value for c in detected_conflicts},
            {c.value for c in case.expected_conflicts},
        )

        i_prec, i_rec, i_f1 = compute_precision_recall_f1(
            set(predicted_intents),
            set(case.expected_intents),
        )

        # Baseline B action metrics & error attribution
        exp_actions = set(case.expected_agent_actions)
        pred_actions = set(predicted_actions)
        act_cov = len(pred_actions.intersection(exp_actions)) / len(exp_actions) if exp_actions else 1.0
        unexp_act_rate = len(pred_actions - exp_actions) / len(pred_actions) if pred_actions else 0.0

        from mosaic.evaluation.error_attribution import determine_error_category
        err_cat = determine_error_category(
            predicted_intents=predicted_intents,
            expected_intents=case.expected_intents,
            predicted_actions=predicted_actions,
            expected_actions=case.expected_agent_actions,
            predicted_verdict=predicted_verdict,
            expected_verdict=case.expected_final_verdict,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        token_usage = _collect_token_usage(*providers)

        return EvaluationSystemResult(
            case_id=case.case_id,
            source_case_id=case.source_case_id,
            system_name=self.system_name,
            predicted_intents=predicted_intents,
            predicted_actions=predicted_actions,
            predicted_verdict=predicted_verdict,
            expected_verdict=case.expected_final_verdict,
            verdict_correct=(predicted_verdict == case.expected_final_verdict),
            intent_precision=i_prec,
            intent_recall=i_rec,
            intent_f1=i_f1,
            action_coverage=round(act_cov, 4),
            unexpected_action_rate=round(unexp_act_rate, 4),
            detected_conflicts=detected_conflicts,
            expected_conflicts=case.expected_conflicts,
            conflict_precision=prec,
            conflict_recall=rec,
            conflict_f1=f1,
            execution_outcome="DIRECT_UNVALIDATED_MULTI_ACTION_SYNTHESIS" if predicted_actions else "NO_INTENT_DETECTED",
            error_category=err_cat,
            latency_ms=round(elapsed_ms, 2),
            token_usage=token_usage,
        )


class MosaicResearchSystem(BaseResearchSystem):
    """MOSAIC Full Architecture System.

    Decomposes multi-intents, executes specialist agents, compiles actions, and passes them through
    the deterministic, state-aware MOSAIC Validation Engine.
    """

    def __init__(
        self,
        orchestrator: Optional[MosaicOrchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or MosaicOrchestrator()

    @property
    def system_name(self) -> str:
        return "mosaic_validated"

    def evaluate_case(self, case: BenchmarkCase) -> EvaluationSystemResult:
        from mosaic.domain.models import PolicyRule

        providers = [
            getattr(self.orchestrator.intake_engine, "llm_provider", None),
            getattr(getattr(self.orchestrator, "compiler", None), "llm_provider", None),
        ]
        _collect_token_usage(*providers)
        start_time = time.perf_counter()

        # Define evaluation policies for security/compliance flags
        eval_policies = [
            PolicyRule(
                id="POL_SUSPICIOUS_LOCATION",
                rule_name="Suspicious Location Escalation",
                description="Require human escalation when suspicious location login flag is present",
                action_types=["restore_login_access"],
                forbidden_active_flags=["SUSPICIOUS_LOCATION_LOGIN"],
                outcome=ValidationVerdict.ESCALATED,
                explanation="Location anomaly requires human verification",
            ),
            PolicyRule(
                id="POL_LEGAL_HOLD",
                rule_name="Legal Hold Escalation",
                description="Require compliance review on legal hold accounts",
                action_types=["restore_login_access"],
                forbidden_active_flags=["PENDING_LEGAL_HOLD"],
                outcome=ValidationVerdict.ESCALATED,
                explanation="Legal hold active",
            ),
            PolicyRule(
                id="POL_MANUAL_AUDIT",
                rule_name="Manual Audit Escalation",
                description="Require audit review when manual audit flag is present",
                action_types=["restore_login_access"],
                forbidden_active_flags=["MANUAL_AUDIT_REQUIRED"],
                outcome=ValidationVerdict.ESCALATED,
                explanation="Manual audit required",
            ),
        ]

        validation_result = self.orchestrator.process_customer_case(
            customer_id="bench_user",
            raw_message=case.customer_message,
            initial_facts=case.initial_facts,
            active_flags=case.active_flags,
            policies=eval_policies,
        )

        # Extract extracted intent names & action types from orchestrator intake
        spans = self.orchestrator.intake_engine.decompose_message(case.customer_message)
        predicted_intents = [s.intent_name for s in spans]

        # Extract action types
        predicted_actions: List[str] = []
        for span in spans:
            # Simple agent lookup matching orchestrator
            agent_map = {
                "restrict_account": "restrict_account",
                "refund_payment": "refund_payment",
                "cancel_subscription": "cancel_subscription",
                "restore_login_access": "restore_login_access",
            }
            if span.intent_name in agent_map:
                predicted_actions.append(agent_map[span.intent_name])

        detected_conflicts = [c.conflict_type for c in validation_result.conflicts]

        prec, rec, f1 = compute_precision_recall_f1(
            {c.value for c in detected_conflicts},
            {c.value for c in case.expected_conflicts},
        )

        i_prec, i_rec, i_f1 = compute_precision_recall_f1(
            set(predicted_intents),
            set(case.expected_intents),
        )

        # MOSAIC action metrics & error attribution
        exp_actions = set(case.expected_agent_actions)
        pred_actions = set(predicted_actions)
        act_cov = len(pred_actions.intersection(exp_actions)) / len(exp_actions) if exp_actions else 1.0
        unexp_act_rate = len(pred_actions - exp_actions) / len(pred_actions) if pred_actions else 0.0

        from mosaic.evaluation.error_attribution import determine_error_category
        err_cat = determine_error_category(
            predicted_intents=predicted_intents,
            expected_intents=case.expected_intents,
            predicted_actions=predicted_actions,
            expected_actions=case.expected_agent_actions,
            predicted_verdict=validation_result.verdict,
            expected_verdict=case.expected_final_verdict,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        token_usage = _collect_token_usage(*providers)

        return EvaluationSystemResult(
            case_id=case.case_id,
            source_case_id=case.source_case_id,
            system_name=self.system_name,
            predicted_intents=predicted_intents,
            predicted_actions=predicted_actions,
            predicted_verdict=validation_result.verdict,
            expected_verdict=case.expected_final_verdict,
            verdict_correct=(validation_result.verdict == case.expected_final_verdict),
            intent_precision=i_prec,
            intent_recall=i_rec,
            intent_f1=i_f1,
            action_coverage=round(act_cov, 4),
            unexpected_action_rate=round(unexp_act_rate, 4),
            detected_conflicts=detected_conflicts,
            expected_conflicts=case.expected_conflicts,
            conflict_precision=prec,
            conflict_recall=rec,
            conflict_f1=f1,
            execution_outcome=f"MOSAIC_VALIDATED_VERDICT_{validation_result.verdict.value}",
            error_category=err_cat,
            latency_ms=round(elapsed_ms, 2),
            token_usage=token_usage,
        )
