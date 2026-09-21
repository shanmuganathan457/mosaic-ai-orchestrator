"""MOSAIC Domain Models.

This module defines Pydantic schemas for the complete lifecycle of customer support cases,
intent spans, sub-tasks, agent proposals, compiled actions, preconditions, postconditions,
policy rules, validation results, conflicts, escalations, response drafts, and audit events.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


# ==========================================
# ENUMS
# ==========================================

class CaseStatus(str, Enum):
    """Lifecycle status of a customer support case."""
    NEW = "NEW"
    ANALYZING = "ANALYZING"
    PENDING_VALIDATION = "PENDING_VALIDATION"
    VALIDATED = "VALIDATED"
    EXECUTED = "EXECUTED"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class IntentCategory(str, Enum):
    """Standardized enterprise customer intent categories."""
    SECURITY = "SECURITY"
    BILLING = "BILLING"
    SUBSCRIPTION = "SUBSCRIPTION"
    ACCESS_RESTORATION = "ACCESS_RESTORATION"
    TECHNICAL_SUPPORT = "TECHNICAL_SUPPORT"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"


class ActionRiskLevel(str, Enum):
    """Risk classification for proposed agent actions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ValidationVerdict(str, Enum):
    """Outcome of the validation process."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ESCALATED = "ESCALATED"


class ConflictType(str, Enum):
    """Classification of validation conflicts."""
    PRECONDITION_UNSATISFIED = "PRECONDITION_UNSATISFIED"
    MUTUALLY_EXCLUSIVE_POSTCONDITION = "MUTUALLY_EXCLUSIVE_POSTCONDITION"
    POLICY_CONSTRAINT_VIOLATION = "POLICY_CONSTRAINT_VIOLATION"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"


# ==========================================
# INTENT & CASE MODELS
# ==========================================

class IntentSpan(BaseModel):
    """Represents a specific extracted intent bounded by exact verbatim text span."""
    id: UUID = Field(default_factory=uuid4)
    category: IntentCategory
    intent_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    verbatim_text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class SubTask(BaseModel):
    """Child task generated from a specific intent span assigned to a worker agent."""
    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    intent_span_id: UUID
    assigned_agent: str
    status: str = "PENDING"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)


class CaseState(BaseModel):
    """Dynamic snapshot of known facts and status regarding a customer case."""
    case_id: UUID
    current_facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value assertions (e.g., {'identity_verified': False, 'payment_verified': True})"
    )
    active_flags: List[str] = Field(
        default_factory=list,
        description="Flags such as ['ACCOUNT_LOCKED', 'FRAUD_ALERT']"
    )
    completed_action_types: List[str] = Field(
        default_factory=list,
        description="List of action types that have already completed in the workflow"
    )
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Case(BaseModel):
    """Top-level customer support interaction case container."""
    id: UUID = Field(default_factory=uuid4)
    customer_id: str
    channel: str = "EMAIL"
    raw_message: str
    status: CaseStatus = CaseStatus.NEW
    intents: List[IntentSpan] = Field(default_factory=list)
    sub_tasks: List[SubTask] = Field(default_factory=list)
    state: CaseState
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==========================================
# AGENT & ACTION MODELS
# ==========================================

class AgentProposal(BaseModel):
    """Unvalidated candidate output produced by an autonomous domain worker agent."""
    id: UUID = Field(default_factory=uuid4)
    sub_task_id: UUID
    agent_name: str
    proposed_intent: str
    natural_language_reasoning: str
    raw_action_payload: Dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Precondition(BaseModel):
    """A required fact state assertion that must evaluate to True before Action execution."""
    fact_key: str
    expected_value: Any
    description: str

    model_config = ConfigDict(frozen=True)


class Postcondition(BaseModel):
    """A target fact state modification produced if Action is successfully executed."""
    fact_key: str
    target_value: Any
    description: str

    model_config = ConfigDict(frozen=True)


class Dependency(BaseModel):
    """Topological or logical dependency required prior to Action approval."""
    prerequisite_action_type: Optional[str] = None
    prerequisite_fact_key: Optional[str] = None
    required_value: Optional[Any] = None
    failure_outcome: ValidationVerdict = ValidationVerdict.BLOCK
    description: str = "Dependency requirement"

    model_config = ConfigDict(frozen=True)


class Action(BaseModel):
    """Compiled machine-checkable action object with explicit state semantics."""
    id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    agent_name: str
    action_type: str
    target_entity_id: str
    risk_level: ActionRiskLevel = ActionRiskLevel.LOW
    preconditions: List[Precondition] = Field(default_factory=list)
    postconditions: List[Postcondition] = Field(default_factory=list)
    dependencies: List[Dependency] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ==========================================
# POLICY & VALIDATION MODELS
# ==========================================

class PolicyRule(BaseModel):
    """Declarative business or safety constraint rule."""
    id: str
    rule_name: str
    description: str
    condition_type: str = Field(
        default="FORBIDDEN_ACTION_STATE",
        description="e.g. FORBIDDEN_ACTION_STATE, MUTUALLY_EXCLUSIVE_ACTIONS, MANDATORY_FLAG_CHECK"
    )
    action_types: List[str] = Field(default_factory=list)
    required_state_facts: Dict[str, Any] = Field(default_factory=dict)
    forbidden_state_facts: Dict[str, Any] = Field(default_factory=dict)
    forbidden_active_flags: List[str] = Field(default_factory=list)
    outcome: ValidationVerdict = ValidationVerdict.BLOCK
    explanation: str = "Policy constraint violated"
    is_active: bool = True


class Conflict(BaseModel):
    """Explicit, machine-readable validation conflict item."""
    conflict_type: ConflictType
    action_id: Optional[UUID] = None
    compounding_action_id: Optional[UUID] = None
    reason_code: str
    message: str
    failed_conditions: List[str] = Field(default_factory=list)
    affected_fact_key: Optional[str] = None
    rule_id: Optional[str] = None
    source: str = "validator"


class ValidationResult(BaseModel):
    """Final decision output from the Validation Engine."""
    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    verdict: ValidationVerdict
    allowed_action_ids: List[UUID] = Field(default_factory=list)
    blocked_action_ids: List[UUID] = Field(default_factory=list)
    escalated_action_ids: List[UUID] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    primary_conflicts: List[Conflict] = Field(default_factory=list)
    secondary_conflicts: List[Conflict] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Escalation(BaseModel):
    """Structured human escalation payload containing context and reasons."""
    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    reason: str
    conflicts: List[Conflict] = Field(default_factory=list)
    primary_conflicts: List[Conflict] = Field(default_factory=list)
    secondary_conflicts: List[Conflict] = Field(default_factory=list)
    suggested_human_actions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResponseDraft(BaseModel):
    """Synthesized final customer response created strictly from allowed actions."""
    case_id: UUID
    content: str
    validated_action_ids: List[UUID]
    is_complete: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditEvent(BaseModel):
    """Immutable audit trail log entry for observability."""
    id: UUID = Field(default_factory=uuid4)
    case_id: UUID
    event_type: str
    actor: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
