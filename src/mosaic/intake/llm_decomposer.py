"""MOSAIC LLM-Backed Intent & Evidence Decomposition Engine.

This module provides an LLM-backed implementation of BaseIntentDecomposer.
It uses BaseLLMProvider to parse raw customer communications into structured IntentSpan objects
with Pydantic boundary validation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

from mosaic.domain.models.schemas import IntentCategory, IntentSpan
from mosaic.intake.decomposer import BaseIntentDecomposer
from mosaic.llm import BaseLLMProvider, LLMError, LLMRequest, LLMResponseError

logger = logging.getLogger("mosaic.intake.llm_decomposer")

INTENT_EXTRACTION_SYSTEM_PROMPT = """You are an expert customer-support intent extraction model for the MOSAIC architecture.
Your sole job is to identify distinct operational intents and verbatim evidence spans in raw customer messages.

Follow these strict rules:
1. Identify ALL distinct customer intents present in the text.
2. Do NOT invent intents not supported by verbatim text evidence in the message.
3. Extract exact verbatim evidence spans from the customer text.
4. Assign an appropriate IntentCategory (SECURITY, BILLING, SUBSCRIPTION, ACCESS_RESTORATION, TECHNICAL_SUPPORT, GENERAL_INQUIRY).
5. Provide a confidence score between 0.0 and 1.0.
6. Do NOT invent entity identifiers (payment_id, account_id, subscription_id) not present in the customer message.
7. Do NOT evaluate business policy rules or validation safety (DO NOT output ALLOW, BLOCK, or ESCALATE).

Return ONLY valid JSON matching the requested schema.
"""


# Boundary Pydantic Schemas for Structured Output Validation
class ExtractedIntentItem(BaseModel):
    category: str = Field(..., description="Intent Category name matching IntentCategory Enum.")
    intent_name: str = Field(..., description="Actionable intent name (e.g. refund_payment, restrict_account).")
    verbatim_text: str = Field(..., description="Exact verbatim text span from customer message.")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedIntentsPayload(BaseModel):
    intents: List[ExtractedIntentItem] = Field(default_factory=list)


INTENT_EXTRACTION_SCHEMA = ExtractedIntentsPayload.model_json_schema()


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
            # 1. Enforce strict IntentCategory enum validation (DO NOT SILENTLY DEFAULT)
            category_str = item.category.upper()
            try:
                category_enum = IntentCategory(category_str)
            except ValueError:
                logger.error("LLM returned unsupported/invalid IntentCategory '%s'.", item.category)
                raise LLMResponseError(
                    f"Invalid IntentCategory '{item.category}' returned by LLM. Must be one of {[c.value for c in IntentCategory]}."
                )

            # 2. Evidence Integrity: Verify verbatim_text exists in raw_message
            match = re.search(re.escape(item.verbatim_text), raw_message, re.IGNORECASE)
            if not match and self.enforce_verbatim_traceability:
                logger.error("LLM returned hallucinated verbatim_text '%s' not present in raw message.", item.verbatim_text)
                raise LLMResponseError(
                    f"Evidence integrity error: verbatim_text '{item.verbatim_text}' is not traceable to customer message."
                )

            start_char = match.start() if match else 0
            end_char = match.end() if match else len(item.verbatim_text)

            # 3. Metadata Provenance: Preserve metadata while deriving explicit IDs from verbatim text if available
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
