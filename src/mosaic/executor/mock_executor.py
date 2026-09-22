"""MOSAIC Mock Backend Executor Implementation.

MOSAIC acts strictly as a governance and validation layer.
The Mock Backend Executor represents downstream execution systems (e.g., Billing Service, Security Service, User Management).

Rules:
- ALLOW: Permitted actions are executed by the mock backend executor.
- BLOCK: Actions are NOT executed.
- ESCALATE: Actions are NOT executed and are marked for human review.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from mosaic.domain.models.schemas import (
    Action,
    ValidationResult,
    ValidationVerdict,
)

logger = logging.getLogger("mosaic.executor")


class ExecutionStatus(str, Enum):
    """Status of an action or case execution attempt."""
    EXECUTED = "EXECUTED"
    BLOCKED = "BLOCKED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ActionExecutionRecord(BaseModel):
    """Record of an individual action's execution attempt."""
    action_id: UUID
    action_type: str
    target_entity_id: str
    agent_name: str
    status: ExecutionStatus
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result_details: Dict[str, Any] = Field(default_factory=dict)
    message: str


class GovernanceExecutionResponse(BaseModel):
    """Complete end-to-end lifecycle response returned by the Phase 8A API."""
    request_id: UUID = Field(default_factory=uuid4)
    customer_id: str
    raw_message: str
    detected_intents: List[Dict[str, Any]]
    proposed_actions: List[Dict[str, Any]]
    current_state: Dict[str, Any]
    primary_conflicts: List[Dict[str, Any]]
    secondary_conflicts: List[Dict[str, Any]]
    final_verdict: ValidationVerdict
    human_readable_explanation: str
    execution_status: ExecutionStatus
    action_execution_records: List[ActionExecutionRecord] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseBackendExecutor(ABC):
    """Abstract interface for backend action execution."""

    @abstractmethod
    def execute_governed_case(
        self,
        customer_id: str,
        raw_message: str,
        detected_intents: List[Any],
        proposed_actions: List[Action],
        current_state: Dict[str, Any],
        validation_result: ValidationResult,
    ) -> GovernanceExecutionResponse:
        """Executes actions based strictly on validation result verdict and allowed action IDs."""
        pass


class MockBackendExecutor(BaseBackendExecutor):
    """Mock implementation of downstream backend systems execution."""

    def execute_governed_case(
        self,
        customer_id: str,
        raw_message: str,
        detected_intents: List[Any],
        proposed_actions: List[Action],
        current_state: Dict[str, Any],
        validation_result: ValidationResult,
    ) -> GovernanceExecutionResponse:
        execution_records: List[ActionExecutionRecord] = []
        action_map = {a.id: a for a in proposed_actions}

        intents_serialized = [
            {
                "id": str(i.id),
                "category": i.category.value if hasattr(i.category, "value") else str(i.category),
                "intent_name": i.intent_name,
                "confidence": i.confidence,
                "verbatim_text": i.verbatim_text,
            }
            for i in detected_intents
        ]

        actions_serialized = [
            {
                "id": str(a.id),
                "action_type": a.action_type,
                "target_entity_id": a.target_entity_id,
                "agent_name": a.agent_name,
                "risk_level": a.risk_level.value if hasattr(a.risk_level, "value") else str(a.risk_level),
            }
            for a in proposed_actions
        ]

        primary_conflicts_serialized = [
            {
                "conflict_type": c.conflict_type.value if hasattr(c.conflict_type, "value") else str(c.conflict_type),
                "reason_code": c.reason_code,
                "message": c.message,
            }
            for c in validation_result.primary_conflicts
        ]

        secondary_conflicts_serialized = [
            {
                "conflict_type": c.conflict_type.value if hasattr(c.conflict_type, "value") else str(c.conflict_type),
                "reason_code": c.reason_code,
                "message": c.message,
            }
            for c in validation_result.secondary_conflicts
        ]

        if validation_result.verdict == ValidationVerdict.ALLOW:
            overall_status = ExecutionStatus.EXECUTED
            explanation = "Validation passed (ALLOW). All proposed actions were safely executed by the mock backend executor."

            for action_id in validation_result.allowed_action_ids:
                action = action_map.get(action_id)
                if action:
                    rec = ActionExecutionRecord(
                        action_id=action.id,
                        action_type=action.action_type,
                        target_entity_id=action.target_entity_id,
                        agent_name=action.agent_name,
                        status=ExecutionStatus.EXECUTED,
                        result_details={"status": "SUCCESS", "simulated_backend": "MockExecutor"},
                        message=f"Action '{action.action_type}' for target '{action.target_entity_id}' successfully executed.",
                    )
                    execution_records.append(rec)

        elif validation_result.verdict == ValidationVerdict.BLOCK:
            overall_status = ExecutionStatus.BLOCKED
            reasons = [c["message"] for c in primary_conflicts_serialized] or ["Deterministic validation rule violation."]
            explanation = f"Validation failed (BLOCK). Execution refused. Reason(s): {'; '.join(reasons)}"

            for action in proposed_actions:
                rec = ActionExecutionRecord(
                    action_id=action.id,
                    action_type=action.action_type,
                    target_entity_id=action.target_entity_id,
                    agent_name=action.agent_name,
                    status=ExecutionStatus.BLOCKED,
                    result_details={"status": "REJECTED_BY_GOVERNANCE"},
                    message=f"Action '{action.action_type}' blocked by MOSAIC governance layer.",
                )
                execution_records.append(rec)

        else:  # ESCALATED
            overall_status = ExecutionStatus.ESCALATED
            reasons = [c["message"] for c in primary_conflicts_serialized] or ["Policy constraint requires human review."]
            explanation = f"Validation paused (ESCALATED). Request flagged for human agent review. Reason(s): {'; '.join(reasons)}"

            for action in proposed_actions:
                rec = ActionExecutionRecord(
                    action_id=action.id,
                    action_type=action.action_type,
                    target_entity_id=action.target_entity_id,
                    agent_name=action.agent_name,
                    status=ExecutionStatus.ESCALATED,
                    result_details={"status": "FLAGGED_FOR_HUMAN_REVIEW"},
                    message=f"Action '{action.action_type}' escalated to human review queue.",
                )
                execution_records.append(rec)

        return GovernanceExecutionResponse(
            customer_id=customer_id,
            raw_message=raw_message,
            detected_intents=intents_serialized,
            proposed_actions=actions_serialized,
            current_state=current_state,
            primary_conflicts=primary_conflicts_serialized,
            secondary_conflicts=secondary_conflicts_serialized,
            final_verdict=validation_result.verdict,
            human_readable_explanation=explanation,
            execution_status=overall_status,
            action_execution_records=execution_records,
        )
