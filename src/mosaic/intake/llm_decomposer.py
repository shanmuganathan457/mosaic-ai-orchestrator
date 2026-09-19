"""MOSAIC LLM-Backed Intent & Evidence Decomposition Engine.

This module provides an LLM-backed implementation of BaseIntentDecomposer.
It uses BaseLLMProvider to parse raw customer communications into structured IntentSpan objects
with Pydantic boundary validation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError

from mosaic.domain.models.schemas import IntentCategory, IntentSpan
from mosaic.intake.decomposer import BaseIntentDecomposer
from mosaic.llm import BaseLLMProvider, LLMError, LLMRequest, LLMResponseError

logger = logging.getLogger("mosaic.intake.llm_decomposer")

# ---------------------------------------------------------------------------
# Canonical vocabularies — single source of truth derived from domain enums
# and the default_agent_map in decomposer.py.
# MUST stay in sync with IntentCategory and default_agent_map.
# ---------------------------------------------------------------------------

# Category Literal — mirrors IntentCategory enum values exactly
CanonicalCategory = Literal[
    "SECURITY",
    "BILLING",
    "SUBSCRIPTION",
    "ACCESS_RESTORATION",
    "TECHNICAL_SUPPORT",
    "GENERAL_INQUIRY",
]

# Intent-name Literal — mirrors default_agent_map keys exactly
CanonicalIntentName = Literal[
    "restrict_account",
    "refund_payment",
    "cancel_subscription",
    "restore_login_access",
]

CANONICAL_INTENT_NAMES = (
    "restrict_account",
    "refund_payment",
    "cancel_subscription",
    "restore_login_access",
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

INTENT_EXTRACTION_SYSTEM_PROMPT = """\
You are a customer-support intent extraction model for the MOSAIC system.
The user will provide a customer message. Extract all distinct intents from it.

== FIELDS ==

category (string enum)
  The broad operational domain. Choose EXACTLY one value from:
    SECURITY | BILLING | SUBSCRIPTION | ACCESS_RESTORATION | TECHNICAL_SUPPORT | GENERAL_INQUIRY

intent_name (string enum)
  The specific canonical action identifier. Choose EXACTLY one value from:
    restrict_account | refund_payment | cancel_subscription | restore_login_access

verbatim_text (string)
  An EXACT contiguous substring copied character-for-character from the customer message.
  SOURCE RULE: This text MUST come from the customer message provided by the user.
               It MUST NOT come from these instructions, field names, or enum values.
  COPY the words the customer actually wrote. Do NOT paraphrase or invent.

confidence (float 0.0–1.0)
  Your extraction confidence.

== EXAMPLE ==

Customer message: "I was charged twice and I cannot log in."

Correct output:
{
  "intents": [
    {
      "category": "BILLING",
      "intent_name": "refund_payment",
      "verbatim_text": "charged twice",
      "confidence": 0.95
    },
    {
      "category": "ACCESS_RESTORATION",
      "intent_name": "restore_login_access",
      "verbatim_text": "cannot log in",
      "confidence": 0.95
    }
  ]
}

== RULES ==
1. Extract ALL distinct intents present in the customer message.
2. verbatim_text must be copied from the customer message — NOT from these instructions.
3. Do not invent entity IDs (pay_xxx, sub_xxx, acc_xxx) that are not in the message.
4. Do not output ALLOW, BLOCK, or ESCALATE.
5. Return ONLY valid JSON.
"""

# ---------------------------------------------------------------------------
# Boundary Pydantic Schemas — enforce canonical vocabularies at LLM boundary
# ---------------------------------------------------------------------------

class ExtractedIntentItem(BaseModel):
    category: CanonicalCategory = Field(
        ...,
        description=(
            "Broad domain category. MUST be exactly one of: "
            "SECURITY, BILLING, SUBSCRIPTION, ACCESS_RESTORATION, "
            "TECHNICAL_SUPPORT, GENERAL_INQUIRY."
        ),
    )
    intent_name: CanonicalIntentName = Field(
        ...,
        description=(
            "Canonical action identifier. MUST be exactly one of: "
            "restrict_account, refund_payment, cancel_subscription, restore_login_access."
        ),
    )
    verbatim_text: str = Field(
        ...,
        description=(
            "Exact contiguous substring copied from the customer message. "
            "Must appear verbatim in the original text."
        ),
    )
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedIntentsPayload(BaseModel):
    intents: List[ExtractedIntentItem] = Field(default_factory=list)


INTENT_EXTRACTION_SCHEMA = ExtractedIntentsPayload.model_json_schema()


# ---------------------------------------------------------------------------
# LLM Decomposer
# ---------------------------------------------------------------------------

class LLMIntentDecomposer(BaseIntentDecomposer):
    """LLM-backed Intent Decomposition Strategy relying on BaseLLMProvider."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        system_prompt: str = INTENT_EXTRACTION_SYSTEM_PROMPT,
        enforce_verbatim_traceability: bool = True,
    ) -> None:
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt
        self.enforce_verbatim_traceability = enforce_verbatim_traceability

    def decompose_message(self, raw_message: str) -> List[IntentSpan]:
        """Parses raw customer text via LLM provider into validated IntentSpan objects."""
        if not raw_message or not raw_message.strip():
            logger.warning("Empty customer message provided to LLMIntentDecomposer.")
            return []

        request = LLMRequest(
            system_prompt=self.system_prompt,
            user_prompt=f"Extract intents from customer message:\n\"\"\"{raw_message}\"\"\"",
            structured_schema=INTENT_EXTRACTION_SCHEMA,
            temperature=0.0,
        )

        try:
            response = self.llm_provider.generate(request)
        except LLMError as e:
            logger.error("LLM provider generation error during intent decomposition: %s", str(e))
            raise LLMResponseError(f"LLM intent decomposition failed: {str(e)}") from e

        # Validate & parse structured response
        raw_output: Any = None

        if response.content:
            try:
                raw_output = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse LLM content as JSON: %s", response.content)
                raise LLMResponseError(f"Malformed non-JSON output from LLM: {str(e)}") from e
        elif response.structured_output is not None:
            raw_output = response.structured_output

        if not isinstance(raw_output, dict):
            raise LLMResponseError("LLM response did not provide a valid JSON dictionary payload.")

        try:
            payload = ExtractedIntentsPayload.model_validate(raw_output)
        except ValidationError as e:
            logger.error("Pydantic validation failed for LLM structured intent payload: %s", str(e))
            raise LLMResponseError(f"Structured output schema validation failed: {str(e)}") from e

        # Map parsed Pydantic items to domain IntentSpan objects with integrity validation
        spans: List[IntentSpan] = []
        for item in payload.intents:
            # 1. Map CanonicalCategory string to IntentCategory domain enum
            # (Pydantic already validated the value is one of the six allowed strings)
            try:
                category_enum = IntentCategory(item.category)
            except ValueError:
                # Should never reach here due to Pydantic Literal validation
                logger.error("Unexpected IntentCategory mapping failure for '%s'.", item.category)
                raise LLMResponseError(
                    f"Invalid IntentCategory '{item.category}' returned by LLM. "
                    f"Must be one of {[c.value for c in IntentCategory]}."
                )

            # 2. Evidence Integrity: Verify verbatim_text is an exact substring of raw_message
            match = re.search(re.escape(item.verbatim_text), raw_message, re.IGNORECASE)
            if not match and self.enforce_verbatim_traceability:
                logger.error(
                    "LLM returned hallucinated verbatim_text '%s' not present in raw message.",
                    item.verbatim_text,
                )
                raise LLMResponseError(
                    f"Evidence integrity error: verbatim_text '{item.verbatim_text}' "
                    f"is not traceable to customer message."
                )

            start_char = match.start() if match else 0
            end_char = match.end() if match else len(item.verbatim_text)

            # 3. Metadata Provenance: Derive entity IDs from verbatim customer text only
            metadata = dict(item.metadata)
            pay_match = re.search(r"pay_\w+", raw_message, re.IGNORECASE)
            if pay_match and "payment_id" not in metadata:
                metadata["payment_id"] = pay_match.group(0)

            sub_match = re.search(r"sub_\w+", raw_message, re.IGNORECASE)
            if sub_match and "subscription_id" not in metadata:
                metadata["subscription_id"] = sub_match.group(0)

            acc_match = re.search(r"acc_\w+", raw_message, re.IGNORECASE)
            if acc_match and "account_id" not in metadata:
                metadata["account_id"] = acc_match.group(0)

            span = IntentSpan(
                category=category_enum,
                intent_name=item.intent_name,
                confidence=item.confidence,
                verbatim_text=item.verbatim_text,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata,
            )
            spans.append(span)

        logger.info("LLMIntentDecomposer successfully extracted %d intent spans.", len(spans))
        return spans
