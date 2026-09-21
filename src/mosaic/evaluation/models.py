"""Evaluation Domain Models & Schemas for Benchmark Dataset and Runner."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

from mosaic.domain.models import Action, Case, ConflictType, SubTask, IntentSpan, ValidationVerdict


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


@dataclass
class SharedEvaluationContext:
    """Shared upstream evaluation artifacts computed once per benchmark case.

    Holds the decomposed case + subtask_mappings from a single LLM intent decomposition call,
    plus a pre-compiled Action map (proposal_id → Action) from a single LLM compilation pass.

    These artifacts are shared across Baseline A, Baseline B, and MOSAIC to eliminate
    redundant LLM calls while preserving fair comparison.

    RESEARCH INTEGRITY NOTE:
    - Ground truth fields (expected_verdict, expected_intents, expected_conflicts) are NEVER stored here.
    - Each system performs its own independent downstream logic after receiving these shared inputs.
    - MOSAIC still runs its full deterministic validation independently.
    """

    case: Case
    """Shared CaseState container built from the single decomposition pass."""

    subtask_mappings: List[Tuple["SubTask", "IntentSpan", str]]
    """Ordered list of (SubTask, IntentSpan, agent_name) tuples from decomposition."""

    compiled_actions: List["Action"]
    """Pre-compiled Action objects from a single compile pass over all proposals."""

    token_usage: Dict[str, Any] = field(default_factory=dict)
    """Aggregated LLM token usage from shared decomposition + compilation calls."""

    llm_call_count: int = 0
    """Number of LLM generate() calls made during shared context construction."""


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
    primary_conflicts: List[ConflictType] = Field(default_factory=list)
    secondary_conflicts: List[ConflictType] = Field(default_factory=list)
    expected_conflicts: List[ConflictType]
    conflict_precision: float
    conflict_recall: float
    conflict_f1: float
    execution_outcome: str
    error_category: str = Field(
        default="CORRECT_ALL",
        description="Error attribution taxonomy category: CORRECT_ALL, INCORRECT_INTENT_EXTRACTION, INCORRECT_ACTION_COMPILATION, INCORRECT_VALIDATION_RESULT, INCORRECT_FINAL_INTERPRETATION."
    )
    latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds for this benchmark case.")
    token_usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage metadata (prompt_tokens, completion_tokens, total_tokens) if available.")

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
    mean_latency_ms: float = Field(default=0.0, description="Mean per-case latency in milliseconds.")
    p95_latency_ms: float = Field(default=0.0, description="95th percentile per-case latency in milliseconds.")
    total_prompt_tokens: Optional[int] = Field(default=None, description="Total LLM prompt tokens across dataset if available.")
    total_completion_tokens: Optional[int] = Field(default=None, description="Total LLM completion tokens across dataset if available.")
    total_tokens: Optional[int] = Field(default=None, description="Total LLM tokens across dataset if available.")

    model_config = ConfigDict(frozen=True)


class FailedCaseRecord(BaseModel):
    """Container recording an unhandled exception during benchmark case execution."""
    case_id: str
    failure_stage: str = Field(default="UNKNOWN", description="Stage where failure occurred: INTENT_DECOMPOSITION, ACTION_COMPILATION, SYSTEM_EVALUATION, etc.")
    exception_type: str = Field(..., description="Exception class name (e.g. LLMError, TimeoutError).")
    error_message: str = Field(..., description="Human-readable exception details.")

    model_config = ConfigDict(frozen=True)


class EvaluationReport(BaseModel):
    """Complete serialized JSON evaluation report output."""
    dataset_version: str
    provider_name: str = "mock"
    model_name: str = "mock-deterministic-v1"
    evaluation_design: str = Field(default="standard", description="Evaluation design methodology: 'standard' or 'shared_context'.")
    total_cases: int
    completed_cases: int = 0
    failed_cases_count: int = 0
    skipped_resumed_cases: int = 0
    successful_llm_calls: int = 0
    failed_llm_calls: int = 0
    systems: Dict[str, AggregateMetrics] = Field(default_factory=dict)
    per_case_results: List[EvaluationSystemResult] = Field(default_factory=list)
    failed_case_records: List[FailedCaseRecord] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)
