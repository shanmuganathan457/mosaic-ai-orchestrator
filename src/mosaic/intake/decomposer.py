"""MOSAIC Deterministic Intent & Evidence Decomposition Engine.

This module provides a rule-based, deterministic intent extractor and case orchestrator
that converts raw customer text communications into structured IntentSpan objects and SubTask records.
"""

import logging
import re
from typing import Any, Dict, List, Tuple
from uuid import UUID

from mosaic.domain.models.schemas import (
    Case,
    CaseState,
    CaseStatus,
    IntentCategory,
    IntentSpan,
    SubTask,
)

logger = logging.getLogger("mosaic.intake.decomposer")


class IntentDecompositionPattern:
    """Pattern specification for rule-based intent matching."""

    def __init__(
        self,
        category: IntentCategory,
        intent_name: str,
        regex_pattern: str,
        assigned_agent: str,
        entity_extractor: Any = None,
    ) -> None:
        self.category = category
        self.intent_name = intent_name
        self.pattern = re.compile(regex_pattern, re.IGNORECASE)
        self.assigned_agent = assigned_agent
        self.entity_extractor = entity_extractor


# Default rule-based pattern registry for deterministic decomposition
DEFAULT_PATTERNS = [
    IntentDecompositionPattern(
        category=IntentCategory.SECURITY,
        intent_name="restrict_account",
        regex_pattern=r"\b(hacked|compromised|unauthorized access|suspicious activity|security breach)\b",
        assigned_agent="SecurityMockAgent",
    ),
    IntentDecompositionPattern(
        category=IntentCategory.BILLING,
        intent_name="refund_payment",
        regex_pattern=r"\b(refund|chargeback|reimburse|wrongly charged)\b",
        assigned_agent="BillingMockAgent",
    ),
    IntentDecompositionPattern(
        category=IntentCategory.SUBSCRIPTION,
        intent_name="cancel_subscription",
        regex_pattern=r"\b(cancel|terminate|stop|end)( my)? (subscription|plan|membership|recurring)\b",
        assigned_agent="SubscriptionMockAgent",
    ),
    IntentDecompositionPattern(
        category=IntentCategory.ACCESS_RESTORATION,
        intent_name="restore_login_access",
        regex_pattern=r"\b(cannot log in|can't login|locked out|restore access|reset password|login access)\b",
        assigned_agent="AccessMockAgent",
    ),
]


class IntentDecompositionEngine:
    """Rule-based, deterministic intent decomposition and sub-task creation engine."""

    def __init__(self, patterns: List[IntentDecompositionPattern] | None = None) -> None:
        self.patterns = patterns or DEFAULT_PATTERNS

    def decompose_message(self, raw_message: str) -> List[IntentSpan]:
        """Parses raw text and extracts non-overlapping, evidence-backed IntentSpan records."""
        spans: List[IntentSpan] = []

        for pat in self.patterns:
            for match in pat.pattern.finditer(raw_message):
                start, end = match.span()
                matched_text = match.group(0)

                # Extract metadata IDs if present in text (e.g., payment_101 or sub_202)
                metadata: Dict[str, Any] = {}
                pay_match = re.search(r"pay_\w+", raw_message, re.IGNORECASE)
                if pay_match:
                    metadata["payment_id"] = pay_match.group(0)

                sub_match = re.search(r"sub_\w+", raw_message, re.IGNORECASE)
                if sub_match:
                    metadata["subscription_id"] = sub_match.group(0)

                acc_match = re.search(r"acc_\w+", raw_message, re.IGNORECASE)
                if acc_match:
                    metadata["account_id"] = acc_match.group(0)

                spans.append(
                    IntentSpan(
                        category=pat.category,
                        intent_name=pat.intent_name,
                        confidence=0.95,
                        verbatim_text=matched_text,
                        start_char=start,
                        end_char=end,
                        metadata=metadata,
                    )
                )

        logger.debug("Decomposed message into %d intent spans.", len(spans))
        return spans

    def create_case_from_intake(
        self,
        customer_id: str,
        raw_message: str,
        initial_facts: Dict[str, Any] | None = None,
        active_flags: List[str] | None = None,
    ) -> Tuple[Case, List[Tuple[SubTask, IntentSpan, str]]]:
        """Creates a canonical Case container, extracts IntentSpans, and generates SubTask assignments."""
        spans = self.decompose_message(raw_message)
        
        # Build CaseState
        state = CaseState(
            case_id=UUID("00000000-0000-0000-0000-000000000000"),  # Temporary until Case ID generated
            current_facts=initial_facts or {},
            active_flags=active_flags or [],
        )

        case = Case(
            customer_id=customer_id,
            raw_message=raw_message,
            status=CaseStatus.ANALYZING,
            intents=spans,
            state=state,
        )

        # Update case_id on state
        case = case.model_copy(update={"state": case.state.model_copy(update={"case_id": case.id})})

        # Generate subtasks and map to target assigned agents
        subtask_mappings: List[Tuple[SubTask, IntentSpan, str]] = []
        sub_tasks: List[SubTask] = []

        for span in spans:
            assigned_agent = "GeneralMockAgent"
            for pat in self.patterns:
                if pat.intent_name == span.intent_name:
                    assigned_agent = pat.assigned_agent
                    break

            subtask = SubTask(
                case_id=case.id,
                intent_span_id=span.id,
                assigned_agent=assigned_agent,
            )
            sub_tasks.append(subtask)
            subtask_mappings.append((subtask, span, assigned_agent))

        case = case.model_copy(update={"sub_tasks": sub_tasks})

        logger.info(
            "Created Case id=%s for customer=%s with %d intents and %d sub-tasks.",
            case.id,
            customer_id,
            len(spans),
            len(sub_tasks),
        )

        return case, subtask_mappings
