"""MOSAIC End-to-End Orchestrator Pipeline.

This module orchestrates the complete deterministic MOSAIC pipeline:
Raw Customer Input -> Intent Intake -> Mock Domain Agents -> Semantic Compiler -> Validation Engine.
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

from mosaic.agents import (
    AccessMockAgent,
    BaseMockAgent,
    BillingMockAgent,
    SecurityMockAgent,
    SubscriptionMockAgent,
)
from mosaic.compiler import SemanticStateCompiler
from mosaic.domain.models import (
    Action,
    Case,
    CaseState,
    PolicyRule,
    ValidationResult,
)
from mosaic.intake.decomposer import IntentDecompositionEngine
from mosaic.validation import DefaultValidationEngine

logger = logging.getLogger("mosaic.orchestrator")


class MosaicOrchestrator:
    """Central pipeline coordinator uniting Intake, Agents, Compiler, and Validation Engine."""

    def __init__(
        self,
        intake_engine: IntentDecompositionEngine | None = None,
        compiler: SemanticStateCompiler | None = None,
        validation_engine: DefaultValidationEngine | None = None,
        agent_registry: Dict[str, BaseMockAgent] | None = None,
    ) -> None:
        self.intake_engine = intake_engine or IntentDecompositionEngine()
        self.compiler = compiler or SemanticStateCompiler()
        self.validation_engine = validation_engine or DefaultValidationEngine()
        
        self.agent_registry = agent_registry or {
            "SecurityMockAgent": SecurityMockAgent(),
            "BillingMockAgent": BillingMockAgent(),
            "SubscriptionMockAgent": SubscriptionMockAgent(),
            "AccessMockAgent": AccessMockAgent(),
        }

    def process_customer_case(
        self,
        customer_id: str,
        raw_message: str,
        initial_facts: Optional[Dict] = None,
        active_flags: Optional[List[str]] = None,
        policies: Optional[List[PolicyRule]] = None,
    ) -> ValidationResult:
        """Processes raw customer communication through complete MOSAIC validation flow.

        Returns:
            ValidationResult containing verdict (ALLOW, BLOCK, ESCALATED), allowed actions,
            blocked actions, and diagnostic conflicts.
        """
        logger.info("Processing case for customer_id=%s...", customer_id)

        # 1. Intake & Intent Decomposition
        case, subtask_mappings = self.intake_engine.create_case_from_intake(
            customer_id=customer_id,
            raw_message=raw_message,
            initial_facts=initial_facts,
            active_flags=active_flags,
        )

        compiled_actions: List[Action] = []

        # 2. Agent Execution -> Proposal Generation
        for subtask, intent_span, agent_name in subtask_mappings:
            agent = self.agent_registry.get(agent_name)
            if not agent:
                logger.warning("No registered agent found for name '%s'. Skipping.", agent_name)
                continue

            proposal = agent.process_subtask(subtask, intent_span)

            # 3. Semantic State Compilation -> Structured Action
            action = self.compiler.compile_proposal(proposal)
            compiled_actions.append(action)

        # 4. Deterministic Validation Engine
        validation_result = self.validation_engine.validate(
            case_state=case.state,
            proposed_actions=compiled_actions,
            policies=policies or [],
        )

        logger.info(
            "Orchestrator completed case_id=%s. Verdict=%s.",
            case.id,
            validation_result.verdict.value,
        )

        return validation_result
