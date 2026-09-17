"""MOSAIC Semantic State Compiler Module.

This module is responsible for compiling unvalidated AgentProposal outputs into
strongly-typed, machine-checkable Action objects with explicit preconditions,
postconditions, dependencies, and risk levels.

The compiler strictly performs translation/formatting and does NOT determine
final validation verdicts (ALLOW, BLOCK, ESCALATE).
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List
from uuid import UUID

from mosaic.domain.models.schemas import (
    Action,
    ActionRiskLevel,
    AgentProposal,
    Dependency,
    Postcondition,
    Precondition,
    ValidationVerdict,
)

logger = logging.getLogger("mosaic.compiler")


class SemanticCompilerError(Exception):
    """Exception raised when an AgentProposal cannot be compiled into a valid Action."""
    pass


class BaseActionCompiler(ABC):
    """Abstract Strategy interface for semantic action compilers."""

    @abstractmethod
    def compile_proposal(self, proposal: AgentProposal) -> Action:
        """Translates an AgentProposal into a strongly-typed Action primitive."""
        pass


class SemanticStateCompiler(BaseActionCompiler):
    """Compiles natural language / semi-structured AgentProposals into structured Action objects (Deterministic Baseline)."""

    def compile_proposal(self, proposal: AgentProposal) -> Action:
        """Translates an AgentProposal into a strongly-typed Action."""
        logger.debug("Compiling AgentProposal id=%s from agent=%s", proposal.id, proposal.agent_name)
        payload = proposal.raw_action_payload

        if "action_type" not in payload:
            raise SemanticCompilerError(f"Proposal {proposal.id} missing mandatory 'action_type' in payload.")

        action_type = payload["action_type"]
        target_entity = payload.get("target_entity", "default_target")

        # Parse Risk Level
        risk_str = str(payload.get("risk_level", "LOW")).upper()
        try:
            risk_level = ActionRiskLevel(risk_str)
        except ValueError:
            risk_level = ActionRiskLevel.LOW

        # Parse Preconditions
        preconditions: List[Precondition] = []
        raw_preconditions = payload.get("preconditions", [])
        for p in raw_preconditions:
            preconditions.append(
                Precondition(
                    fact_key=p["fact_key"],
                    expected_value=p["expected_value"],
                    description=p.get("description", f"Requires {p['fact_key']}={p['expected_value']}")
                )
            )

        # Parse Postconditions
        postconditions: List[Postcondition] = []
        raw_postconditions = payload.get("postconditions", [])
        for p in raw_postconditions:
            postconditions.append(
                Postcondition(
                    fact_key=p["fact_key"],
                    target_value=p["target_value"],
                    description=p.get("description", f"Sets {p['fact_key']}={p['target_value']}")
                )
            )

        # Parse Dependencies
        dependencies: List[Dependency] = []
        raw_dependencies = payload.get("parameters", {}).get("dependencies", payload.get("dependencies", []))
        for d in raw_dependencies:
            failure_str = str(d.get("failure_outcome", "BLOCK")).upper()
            try:
                failure_outcome = ValidationVerdict(failure_str)
            except ValueError:
                failure_outcome = ValidationVerdict.BLOCK

            dependencies.append(
                Dependency(
                    prerequisite_action_type=d.get("prerequisite_action_type"),
                    prerequisite_fact_key=d.get("prerequisite_fact_key"),
                    required_value=d.get("required_value"),
                    failure_outcome=failure_outcome,
                    description=d.get("description", "Dependency requirement")
                )
            )

        # Build Action
        compiled_action = Action(
            proposal_id=proposal.id,
            agent_name=proposal.agent_name,
            action_type=action_type,
            target_entity_id=target_entity,
            risk_level=risk_level,
            preconditions=preconditions,
            postconditions=postconditions,
            dependencies=dependencies,
            parameters=payload.get("parameters", {})
        )

        logger.info(
            "Successfully compiled Action id=%s (type=%s, agent=%s) with %d preconditions, %d postconditions, %d dependencies.",
            compiled_action.id,
            compiled_action.action_type,
            compiled_action.agent_name,
            len(compiled_action.preconditions),
            len(compiled_action.postconditions),
            len(compiled_action.dependencies),
        )

        return compiled_action
