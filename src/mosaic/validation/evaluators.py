"""MOSAIC Sub-Evaluators for Deterministic Validation.

This module provides focused, modular evaluation components for:
1. Precondition Evaluation
2. Dependency Evaluation
3. Policy Constraint Evaluation
4. Cross-Action / Postcondition Conflict Evaluation
"""

import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from mosaic.domain.models.schemas import (
    Action,
    CaseState,
    Conflict,
    ConflictType,
    PolicyRule,
    ValidationVerdict,
)

logger = logging.getLogger("mosaic.validation.evaluators")


class PreconditionEvaluator:
    """Evaluates whether an action's required state preconditions are satisfied by current case facts."""

    def evaluate_action(
        self, action: Action, case_state: CaseState
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        for prec in action.preconditions:
            fact_key = prec.fact_key
            expected = prec.expected_value
            
            # Missing fact key entirely
            if fact_key not in case_state.current_facts:
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.PRECONDITION_UNSATISFIED,
                        action_id=action.id,
                        reason_code="MISSING_STATE_FACT",
                        message=f"Action '{action.action_type}' requires fact '{fact_key}' which is unknown/missing in case state.",
                        failed_conditions=[f"{fact_key} == {expected} (Fact missing)"],
                        affected_fact_key=fact_key,
                        source="evaluator/precondition",
                    )
                )
            else:
                actual = case_state.current_facts[fact_key]
                if actual != expected:
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.PRECONDITION_UNSATISFIED,
                            action_id=action.id,
                            reason_code="PRECONDITION_FAILED",
                            message=f"Action '{action.action_type}' requires {fact_key}={expected}, but current state is {actual}.",
                            failed_conditions=[f"{fact_key} == {expected} (Actual: {actual})"],
                            affected_fact_key=fact_key,
                            source="evaluator/precondition",
                        )
                    )
        return conflicts


class DependencyEvaluator:
    """Evaluates explicit action prerequisites against current facts or completed workflow actions."""

    def evaluate_action(
        self, action: Action, case_state: CaseState
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        for dep in action.dependencies:
            # Fact-based prerequisite
            if dep.prerequisite_fact_key is not None:
                fact_key = dep.prerequisite_fact_key
                expected = dep.required_value
                actual = case_state.current_facts.get(fact_key)
                
                if actual != expected:
                    conflict_type = (
                        ConflictType.AMBIGUOUS_EVIDENCE
                        if dep.failure_outcome == ValidationVerdict.ESCALATED
                        else ConflictType.MISSING_DEPENDENCY
                    )
                    conflicts.append(
                        Conflict(
                            conflict_type=conflict_type,
                            action_id=action.id,
                            reason_code="DEPENDENCY_NOT_MET",
                            message=f"Action '{action.action_type}' depends on '{fact_key}' reaching '{expected}'. Current state is '{actual}'. {dep.description}",
                            failed_conditions=[f"dependency({fact_key} == {expected})"],
                            affected_fact_key=fact_key,
                            source="evaluator/dependency",
                        )
                    )
            
            # Completed action-type prerequisite
            if dep.prerequisite_action_type is not None:
                prereq_type = dep.prerequisite_action_type
                if prereq_type not in case_state.completed_action_types:
                    conflict_type = (
                        ConflictType.AMBIGUOUS_EVIDENCE
                        if dep.failure_outcome == ValidationVerdict.ESCALATED
                        else ConflictType.MISSING_DEPENDENCY
                    )
                    conflicts.append(
                        Conflict(
                            conflict_type=conflict_type,
                            action_id=action.id,
                            reason_code="DEPENDENCY_ACTION_UNCOMPLETED",
                            message=f"Action '{action.action_type}' depends on prior completion of action '{prereq_type}'. {dep.description}",
                            failed_conditions=[f"prerequisite_action('{prereq_type}')"],
                            source="evaluator/dependency",
                        )
                    )
        return conflicts


class PolicyEvaluator:
    """Evaluates declarative business policies against actions and current case state/flags."""

    def evaluate_action(
        self, action: Action, case_state: CaseState, policies: List[PolicyRule]
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        for policy in policies:
            if not policy.is_active:
                continue
            
            # Check if this policy targets the given action type
            if policy.action_types and action.action_type not in policy.action_types:
                continue

            # Check forbidden active flags (e.g., ACCOUNT_RESTRICTED)
            for flag in policy.forbidden_active_flags:
                if flag in case_state.active_flags:
                    conflict_type = (
                        ConflictType.AMBIGUOUS_EVIDENCE
                        if policy.outcome == ValidationVerdict.ESCALATED
                        else ConflictType.POLICY_CONSTRAINT_VIOLATION
                    )
                    conflicts.append(
                        Conflict(
                            conflict_type=conflict_type,
                            action_id=action.id,
                            reason_code="POLICY_FORBIDDEN_FLAG",
                            message=f"Action '{action.action_type}' violates policy '{policy.id}' ({policy.rule_name}): Case has active flag '{flag}'. {policy.explanation}",
                            failed_conditions=[f"forbidden_flag('{flag}')"],
                            rule_id=policy.id,
                            source="evaluator/policy",
                        )
                    )

            # Check required state facts for policy
            for req_key, req_val in policy.required_state_facts.items():
                actual_val = case_state.current_facts.get(req_key)
                if actual_val != req_val:
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.POLICY_CONSTRAINT_VIOLATION,
                            action_id=action.id,
                            reason_code="POLICY_REQUIRED_FACT_FAILED",
                            message=f"Action '{action.action_type}' violates policy '{policy.id}' ({policy.rule_name}): Requires {req_key}={req_val}, but actual is {actual_val}. {policy.explanation}",
                            failed_conditions=[f"policy_required({req_key} == {req_val})"],
                            affected_fact_key=req_key,
                            rule_id=policy.id,
                            source="evaluator/policy",
                        )
                    )

            # Check forbidden state facts for policy
            for forb_key, forb_val in policy.forbidden_state_facts.items():
                actual_val = case_state.current_facts.get(forb_key)
                if actual_val == forb_val:
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.POLICY_CONSTRAINT_VIOLATION,
                            action_id=action.id,
                            reason_code="POLICY_FORBIDDEN_FACT_PRESENT",
                            message=f"Action '{action.action_type}' violates policy '{policy.id}' ({policy.rule_name}): State fact {forb_key}={forb_val} is forbidden. {policy.explanation}",
                            failed_conditions=[f"policy_forbidden({forb_key} != {forb_val})"],
                            affected_fact_key=forb_key,
                            rule_id=policy.id,
                            source="evaluator/policy",
                        )
                    )

        return conflicts


class PostconditionConflictEvaluator:
    """Detects cross-action conflicts where combined postconditions produce mutually exclusive target facts."""

    def evaluate_cross_actions(
        self, actions: List[Action]
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        # Map fact_key -> List of (action_id, action_type, target_value)
        target_fact_map: Dict[str, List[Tuple[UUID, str, str]]] = {}

        for act in actions:
            for post in act.postconditions:
                key = post.fact_key
                val = str(post.target_value)
                if key not in target_fact_map:
                    target_fact_map[key] = []
                target_fact_map[key].append((act.id, act.action_type, val))

        # Check for conflicting postcondition target values on the same fact key
        for key, targets in target_fact_map.items():
            if len(targets) > 1:
                # Compare all pairs
                for i in range(len(targets)):
                    for j in range(i + 1, len(targets)):
                        act_id1, act_type1, val1 = targets[i]
                        act_id2, act_type2, val2 = targets[j]
                        if val1 != val2:
                            conflicts.append(
                                Conflict(
                                    conflict_type=ConflictType.MUTUALLY_EXCLUSIVE_POSTCONDITION,
                                    action_id=act_id1,
                                    compounding_action_id=act_id2,
                                    reason_code="MUTUALLY_EXCLUSIVE_POSTCONDITIONS",
                                    message=f"Action '{act_type1}' target postcondition {key}='{val1}' conflicts with Action '{act_type2}' target postcondition {key}='{val2}'.",
                                    failed_conditions=[f"{key}: '{val1}' vs '{val2}'"],
                                    affected_fact_key=key,
                                    source="evaluator/cross_action",
                                )
                            )
        return conflicts
