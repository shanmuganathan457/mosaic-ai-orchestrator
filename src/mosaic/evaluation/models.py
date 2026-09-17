"""Evaluation Domain Models & Schemas for Benchmark Dataset and Runner."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from mosaic.domain.models import ValidationVerdict, ConflictType


class BenchmarkCase(BaseModel):
    """Strongly typed Ground Truth container for a research benchmark case."""
    case_id: str = Field(..., description="Unique human-readable benchmark case identifier (e.g., 'case_001').")
    source_case_id: Optional[str] = Field(default=None, description="Reference ID to authoritative source case in v1 dataset.")
    category: str = Field(..., description="Benchmark scenario category (e.g., 'independent_valid', 'cross_action_conflict').")
    customer_message: str = Field(..., description="Raw verbatim customer text input.")
    expected_intents: List[str] = Field(..., description="List of expected intent names extracted.")
    expected_intent_evidence: Dict[str, str] = Field(default_factory=dict, description="Verbatim evidence map per intent.")
    initial_facts: Dict[str, Any] = Field(default_factory=dict, description="Initial case state facts.")
    active_flags: List[str] = Field(default_factory=list, description="Initial active case flags.")
    expected_agent_actions: List[str] = Field(..., description="Expected compiled action types.")
    expected_final_verdict: ValidationVerdict = Field(..., description="Ground truth validation verdict (ALLOW, BLOCK, ESCALATED).")
    expected_conflicts: List[ConflictType] = Field(default_factory=list, description="Expected conflict types if BLOCK or ESCALATED.")
    expected_escalation_reason: Optional[str] = Field(default=None, description="Human-readable rationale for escalation/conflict.")
    rationale: str = Field(..., description="Human-readable rationale explaining the expected verdict.")

    model_config = ConfigDict(frozen=True)


class EvaluationSystemResult(BaseModel):
    """Structured execution output for a single system operating on a BenchmarkCase."""
    case_id: str
    source_case_id: Optional[str] = None
    system_name: str
    predicted_intents: List[str]
    predicted_actions: List[str]
    predicted_verdict: ValidationVerdict
    expected_verdict: ValidationVerdict
    verdict_correct: bool
    intent_precision: float
    intent_recall: float
    intent_f1: float
    action_coverage: float = Field(default=1.0, description="Fraction of expected actions covered by predicted actions.")
    unexpected_action_rate: float = Field(default=0.0, description="Fraction of predicted actions not expected.")
    detected_conflicts: List[ConflictType]
    expected_conflicts: List[ConflictType]
    conflict_precision: float
    conflict_recall: float
    conflict_f1: float
    execution_outcome: str
    error_category: str = Field(
        default="CORRECT_ALL",
        description="Error attribution taxonomy category: CORRECT_ALL, INCORRECT_INTENT_EXTRACTION, INCORRECT_ACTION_COMPILATION, INCORRECT_VALIDATION_RESULT, INCORRECT_FINAL_INTERPRETATION."
    )

    model_config = ConfigDict(frozen=True)


class AggregateMetrics(BaseModel):
    """Aggregated evaluation metrics for a specific system across the dataset."""
    total_cases: int
    verdict_accuracy: float
    intent_precision: float
    intent_recall: float
    intent_f1: float
    action_coverage: float = 1.0
    unexpected_action_rate: float = 0.0
    conflict_precision: float
    conflict_recall: float
    conflict_f1: float
    escalation_precision: float
    false_escalation_rate: float
    error_attribution_counts: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class EvaluationReport(BaseModel):
    """Complete serialized JSON evaluation report output."""
    dataset_version: str
    provider_name: str = "mock"
    model_name: str = "mock-deterministic-v1"
    total_cases: int
    systems: Dict[str, AggregateMetrics]
    per_case_results: List[EvaluationSystemResult]

    model_config = ConfigDict(frozen=True)
