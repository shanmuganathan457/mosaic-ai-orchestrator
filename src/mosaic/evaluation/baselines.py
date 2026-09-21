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
    PolicyRule,
    ValidationResult,
    ValidationVerdict,
)
from mosaic.evaluation.models import BenchmarkCase, EvaluationSystemResult, SharedEvaluationContext
from mosaic.evaluation.metrics import compute_precision_recall_f1
from mosaic.intake.decomposer import BaseIntentDecomposer, IntentDecompositionEngine
from mosaic.llm.base import BaseLLMProvider
from mosaic.orchestrator import MosaicOrchestrator


class SharedContextBuilder:
    """Builds a SharedEvaluationContext for one benchmark case via a single LLM decomposition
    pass and a single LLM compilation pass per proposal.

    This is the ONLY place LLM calls happen under Design C. All three evaluation systems
    (Baseline A, Baseline B, MOSAIC) receive the same pre-computed artifacts.

    Research Integrity Guarantees:
    - Ground truth fields are never passed to this builder.
    - Each system still applies its own independent routing, filtering, and validation logic.
    - MOSAIC runs the full deterministic Validation Engine independently.
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

    def build(
        self,
        customer_message: str,
        initial_facts: Optional[Dict[str, Any]] = None,
        active_flags: Optional[List[str]] = None,
    ) -> SharedEvaluationContext:
        """Executes ONE intent decomposition and ONE compilation pass per proposal.

        Returns a SharedEvaluationContext containing:
        - case: Case object from decomposition
        - subtask_mappings: [(SubTask, IntentSpan, agent_name), ...]
        - compiled_actions: [Action, ...] compiled once per proposal in order
        - token_usage: aggregated LLM usage from decomposition + all compilations
        - llm_call_count: total LLM generate() calls made
        """
        providers = [
            getattr(self.intake_engine, "llm_provider", None),
            getattr(self.compiler, "llm_provider", None),
        ]
        # Flush any stale recorded usage before building
        _collect_token_usage(*providers)

        # ONE intent decomposition
        case, subtask_mappings = self.intake_engine.create_case_from_intake(
            customer_id="bench_user",
            raw_message=customer_message,
            initial_facts=initial_facts or {},
            active_flags=active_flags or [],
        )

        # ONE compile pass per agent proposal
        compiled_actions: List[Action] = []
        for subtask, span, agent_name in subtask_mappings:
            agent = self.agent_registry.get(agent_name)
            if agent:
                proposal = agent.process_subtask(subtask, span)
                action = self.compiler.compile_proposal(proposal)
                compiled_actions.append(action)

        # Collect aggregated token usage from decomposition + all compilations
        token_usage = _collect_token_usage(*providers)

        # Count LLM calls: 1 decomposition + N compilations (one per proposal)
        llm_call_count = 1 + len(compiled_actions)

        return SharedEvaluationContext(
            case=case,
            subtask_mappings=subtask_mappings,
            compiled_actions=compiled_actions,
            token_usage=token_usage,
            llm_call_count=llm_call_count,
        )


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

    @abstractmethod
    def evaluate_case_with_shared_context(
        self,
        case: BenchmarkCase,
        shared: SharedEvaluationContext,
    ) -> EvaluationSystemResult:
        """Evaluates a single benchmark case using pre-built SharedEvaluationContext.

        The shared context provides pre-decomposed intents and pre-compiled actions.
        Each system applies its own downstream logic independently:
        - Baseline A: uses only first intent and first compiled action.
        - Baseline B: uses all compiled actions, ALLOWs without validation.
        - MOSAIC: uses all compiled actions, runs full deterministic validation.

        RESEARCH INTEGRITY: Ground truth fields from BenchmarkCase are read only for
        metric scoring at the END of evaluation — never during decision logic.
        """
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
        primary_conflicts: List[ConflictType] = []
        secondary_conflicts: List[ConflictType] = []

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

    def evaluate_case_with_shared_context(
        self,
        case: BenchmarkCase,
        shared: SharedEvaluationContext,
    ) -> EvaluationSystemResult:
        """Baseline A: uses ONLY first shared intent and first compiled action."""
        start_time = time.perf_counter()

        # Baseline A: select only the FIRST intent from shared decomposition
        predicted_intents: List[str] = []
        predicted_actions: List[str] = []

        if shared.subtask_mappings and shared.compiled_actions:
            _, first_span, _ = shared.subtask_mappings[0]
            predicted_intents.append(first_span.intent_name)
            # First compiled action corresponds to the first subtask
            predicted_actions.append(shared.compiled_actions[0].action_type)

        # Baseline A: ALLOW if action produced, else ESCALATED (no validation)
        predicted_verdict = ValidationVerdict.ALLOW if predicted_actions else ValidationVerdict.ESCALATED
        detected_conflicts: List[ConflictType] = []
        primary_conflicts: List[ConflictType] = []
        secondary_conflicts: List[ConflictType] = []

        prec, rec, f1 = compute_precision_recall_f1(
            {c.value for c in detected_conflicts},
            {c.value for c in case.expected_conflicts},
        )
        i_prec, i_rec, i_f1 = compute_precision_recall_f1(
            set(predicted_intents),
            set(case.expected_intents),
        )

        exp_actions = set(case.expected_agent_actions)
        pred_actions_set = set(predicted_actions)
        act_cov = len(pred_actions_set.intersection(exp_actions)) / len(exp_actions) if exp_actions else 1.0
        unexp_act_rate = len(pred_actions_set - exp_actions) / len(pred_actions_set) if pred_actions_set else 0.0

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
            token_usage=shared.token_usage,
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
        primary_conflicts: List[ConflictType] = []
        secondary_conflicts: List[ConflictType] = []

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

    def evaluate_case_with_shared_context(
        self,
        case: BenchmarkCase,
        shared: SharedEvaluationContext,
    ) -> EvaluationSystemResult:
        """Baseline B: uses ALL shared compiled actions, ALLOWs without validation."""
        start_time = time.perf_counter()

        # Baseline B: use ALL intents and ALL compiled actions from shared context
        predicted_intents = [span.intent_name for _, span, _ in shared.subtask_mappings]
        predicted_actions = [action.action_type for action in shared.compiled_actions]

        # Baseline B: directly ALLOW all compiled actions — no validation
        predicted_verdict = ValidationVerdict.ALLOW if predicted_actions else ValidationVerdict.ESCALATED
        detected_conflicts: List[ConflictType] = []
        primary_conflicts: List[ConflictType] = []
        secondary_conflicts: List[ConflictType] = []

        prec, rec, f1 = compute_precision_recall_f1(
            {c.value for c in detected_conflicts},
            {c.value for c in case.expected_conflicts},
        )
        i_prec, i_rec, i_f1 = compute_precision_recall_f1(
            set(predicted_intents),
            set(case.expected_intents),
        )

        exp_actions = set(case.expected_agent_actions)
        pred_actions_set = set(predicted_actions)
        act_cov = len(pred_actions_set.intersection(exp_actions)) / len(exp_actions) if exp_actions else 1.0
        unexp_act_rate = len(pred_actions_set - exp_actions) / len(pred_actions_set) if pred_actions_set else 0.0

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
            token_usage=shared.token_usage,
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
        primary_conflicts = [c.conflict_type for c in validation_result.primary_conflicts]
        secondary_conflicts = [c.conflict_type for c in validation_result.secondary_conflicts]

        prec, rec, f1 = compute_precision_recall_f1(
            {c.value for c in primary_conflicts},
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

    def evaluate_case_with_shared_context(
        self,
        case: BenchmarkCase,
        shared: SharedEvaluationContext,
    ) -> EvaluationSystemResult:
        """MOSAIC: uses all shared compiled actions, runs full deterministic Validation Engine."""
        start_time = time.perf_counter()

        # Build evaluation policies (same as standalone evaluate_case)
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

        # MOSAIC: run the full deterministic Validation Engine on the SHARED compiled actions
        # This is independent — MOSAIC does NOT shortcut validation because artifacts were shared
        validation_result = self.orchestrator.validation_engine.validate(
            case_state=shared.case.state,
            proposed_actions=shared.compiled_actions,
            policies=eval_policies,
        )

        # Intents and actions from shared context
        predicted_intents = [span.intent_name for _, span, _ in shared.subtask_mappings]
        predicted_actions = [action.action_type for action in shared.compiled_actions]

        detected_conflicts = [c.conflict_type for c in validation_result.conflicts]
        primary_conflicts = [c.conflict_type for c in validation_result.primary_conflicts]
        secondary_conflicts = [c.conflict_type for c in validation_result.secondary_conflicts]

        prec, rec, f1 = compute_precision_recall_f1(
            {c.value for c in primary_conflicts},
            {c.value for c in case.expected_conflicts},
        )
        i_prec, i_rec, i_f1 = compute_precision_recall_f1(
            set(predicted_intents),
            set(case.expected_intents),
        )

        exp_actions = set(case.expected_agent_actions)
        pred_actions_set = set(predicted_actions)
        act_cov = len(pred_actions_set.intersection(exp_actions)) / len(exp_actions) if exp_actions else 1.0
        unexp_act_rate = len(pred_actions_set - exp_actions) / len(pred_actions_set) if pred_actions_set else 0.0

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
            token_usage=shared.token_usage,
        )
