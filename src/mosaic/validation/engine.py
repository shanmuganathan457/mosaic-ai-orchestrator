"""MOSAIC Deterministic Validation Engine Implementation.

This module implements DefaultValidationEngine, orchestrating modular sub-evaluators:
PreconditionEvaluator, DependencyEvaluator, PolicyEvaluator, and PostconditionConflictEvaluator.
"""

import logging
from typing import List, Set
from uuid import UUID

from mosaic.domain.models.schemas import (
    Action,
    CaseState,
    Conflict,
    ConflictType,
    PolicyRule,
    ValidationResult,
    ValidationVerdict,
)
from mosaic.validation.base import BaseValidationEngine
from mosaic.validation.evaluators import (
    DependencyEvaluator,
    PolicyEvaluator,
    PostconditionConflictEvaluator,
    PreconditionEvaluator,
)

logger = logging.getLogger("mosaic.validation.engine")


class DefaultValidationEngine(BaseValidationEngine):
    """Deterministic, non-LLM validation engine for proposed cross-agent actions."""

    def __init__(
        self,
        precondition_evaluator: PreconditionEvaluator | None = None,
        dependency_evaluator: DependencyEvaluator | None = None,
        policy_evaluator: PolicyEvaluator | None = None,
        cross_action_evaluator: PostconditionConflictEvaluator | None = None,
    ) -> None:
        self.precondition_evaluator = precondition_evaluator or PreconditionEvaluator()
        self.dependency_evaluator = dependency_evaluator or DependencyEvaluator()
        self.policy_evaluator = policy_evaluator or PolicyEvaluator()
        self.cross_action_evaluator = cross_action_evaluator or PostconditionConflictEvaluator()

    def validate(
        self,
        case_state: CaseState,
        proposed_actions: List[Action],
        policies: List[PolicyRule],
    ) -> ValidationResult:
        """Evaluates proposed actions against case state, dependencies, policies, and cross-action conflicts."""
        logger.info(
            "Starting validation for case_id=%s with %d proposed actions and %d active policies.",
            case_state.case_id,
            len(proposed_actions),
            len(policies),
        )

        all_conflicts: List[Conflict] = []
        blocked_action_ids: Set[UUID] = set()
        escalated_action_ids: Set[UUID] = set()
        allowed_action_ids: Set[UUID] = set()

        # Step 1: Evaluate per-action preconditions, dependencies, and policies
        for action in proposed_actions:
            action_conflicts: List[Conflict] = []

            # 1a. Preconditions
            prec_conflicts = self.precondition_evaluator.evaluate_action(action, case_state)
            action_conflicts.extend(prec_conflicts)

            # 1b. Dependencies
            dep_conflicts = self.dependency_evaluator.evaluate_action(action, case_state)
            action_conflicts.extend(dep_conflicts)

            # 1c. Policies
            pol_conflicts = self.policy_evaluator.evaluate_action(action, case_state, policies)
            action_conflicts.extend(pol_conflicts)

            if action_conflicts:
                all_conflicts.extend(action_conflicts)
                # Check if any conflict requires ESCALATE
                has_escalation = any(
                    c.conflict_type == ConflictType.AMBIGUOUS_EVIDENCE for c in action_conflicts
                )
                if has_escalation:
                    escalated_action_ids.add(action.id)
                else:
                    blocked_action_ids.add(action.id)
            else:
                allowed_action_ids.add(action.id)

        # Step 2: Cross-action / Postcondition conflict evaluation across candidate allowed actions
        candidate_actions = [a for a in proposed_actions if a.id in allowed_action_ids]
        cross_conflicts = self.cross_action_evaluator.evaluate_cross_actions(candidate_actions)

        if cross_conflicts:
            all_conflicts.extend(cross_conflicts)
            for c in cross_conflicts:
                if c.action_id and c.action_id in allowed_action_ids:
                    allowed_action_ids.remove(c.action_id)
                    blocked_action_ids.add(c.action_id)
                if c.compounding_action_id and c.compounding_action_id in allowed_action_ids:
                    allowed_action_ids.remove(c.compounding_action_id)
                    blocked_action_ids.add(c.compounding_action_id)

        # Step 3: Determine overall Verdict
        if escalated_action_ids:
            overall_verdict = ValidationVerdict.ESCALATED
        elif blocked_action_ids:
            overall_verdict = ValidationVerdict.BLOCK
        else:
            overall_verdict = ValidationVerdict.ALLOW

        result = ValidationResult(
            case_id=case_state.case_id,
            verdict=overall_verdict,
            allowed_action_ids=list(allowed_action_ids),
            blocked_action_ids=list(blocked_action_ids),
            escalated_action_ids=list(escalated_action_ids),
            conflicts=all_conflicts,
        )

        logger.info(
            "Validation finished for case_id=%s. Verdict=%s, Allowed=%d, Blocked=%d, Escalated=%d",
            case_state.case_id,
            overall_verdict.value,
            len(result.allowed_action_ids),
            len(result.blocked_action_ids),
            len(result.escalated_action_ids),
        )

        return result
