"""Research Baseline Systems (Baseline A, Baseline B, and MOSAIC)."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
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
from mosaic.orchestrator import MosaicOrchestrator


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

        return EvaluationSystemResult(
            case_id=case.case_id,
            system_name=self.system_name,
            predicted_intents=predicted_intents,
            predicted_actions=predicted_actions,
            predicted_verdict=predicted_verdict,
            expected_verdict=case.expected_final_verdict,
            verdict_correct=(predicted_verdict == case.expected_final_verdict),
            detected_conflicts=detected_conflicts,
            expected_conflicts=case.expected_conflicts,
            conflict_precision=prec,
            conflict_recall=rec,
            conflict_f1=f1,
            execution_outcome="DIRECT_SINGLE_ACTION_EXECUTION" if predicted_actions else "NO_INTENT_DETECTED",
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
        case_obj, subtask_mappings = self.intake_engine.create_create_case_from_intake(
            customer_id="bench_user",
            raw_message=case.customer_message,
            initial_facts=case.initial_facts,
            active_flags=case.active_flags,
        ) if hasattr(self.intake_engine, "create_create_case_from_intake") else self.intake_engine.create_case_from_intake(
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

        return EvaluationSystemResult(
            case_id=case.case_id,
            system_name=self.system_name,
            predicted_intents=predicted_intents,
            predicted_actions=predicted_actions,
            predicted_verdict=predicted_verdict,
            expected_verdict=case.expected_final_verdict,
            verdict_correct=(predicted_verdict == case.expected_final_verdict),
            detected_conflicts=detected_conflicts,
            expected_conflicts=case.expected_conflicts,
            conflict_precision=prec,
            conflict_recall=rec,
            conflict_f1=f1,
            execution_outcome="DIRECT_UNVALIDATED_MULTI_ACTION_SYNTHESIS" if predicted_actions else "NO_INTENT_DETECTED",
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
        validation_result = self.orchestrator.process_customer_case(
            customer_id="bench_user",
            raw_message=case.customer_message,
            initial_facts=case.initial_facts,
            active_flags=case.active_flags,
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

        return EvaluationSystemResult(
            case_id=case.case_id,
            system_name=self.system_name,
            predicted_intents=predicted_intents,
            predicted_actions=predicted_actions,
            predicted_verdict=validation_result.verdict,
            expected_verdict=case.expected_final_verdict,
            verdict_correct=(validation_result.verdict == case.expected_final_verdict),
            detected_conflicts=detected_conflicts,
            expected_conflicts=case.expected_conflicts,
            conflict_precision=prec,
            conflict_recall=rec,
            conflict_f1=f1,
            execution_outcome=f"MOSAIC_VALIDATED_VERDICT_{validation_result.verdict.value}",
        )
