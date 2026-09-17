"""MOSAIC LLM-Backed Semantic Action Compiler.

This module provides LLMActionCompiler, an LLM-backed implementation of BaseActionCompiler.
It parses AgentProposals into strongly-typed domain Action objects via BaseLLMProvider.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

from mosaic.compiler.state_compiler import BaseActionCompiler, SemanticCompilerError
from mosaic.domain.models.schemas import (
    Action,
    ActionRiskLevel,
    AgentProposal,
    Dependency,
    Postcondition,
    Precondition,
    ValidationVerdict,
)
from mosaic.llm import BaseLLMProvider, LLMError, LLMRequest, LLMResponseError

logger = logging.getLogger("mosaic.compiler.llm_compiler")

ACTION_COMPILATION_SYSTEM_PROMPT = """You are an expert semantic compiler for the MOSAIC customer-support architecture.
Your job is to translate an unvalidated AgentProposal into a strongly-typed, structured Action object.

Follow these strict rules:
1. Identify the core action_type, target_entity_id, and risk_level (LOW, MEDIUM, HIGH, CRITICAL).
2. Extract required preconditions (fact_key, expected_value, description).
3. Extract target postconditions (fact_key, target_value, description).
4. Extract dependencies if present in the proposal.
5. Do NOT invent state facts, payment_id, account_id, or subscription_id not present in the proposal.
6. Do NOT make validation safety decisions (DO NOT output ALLOW, BLOCK, or ESCALATE).

Return ONLY valid JSON matching the requested schema.
"""


# Boundary Pydantic Schemas for LLM Action Compilation
class LLMPreconditionItem(BaseModel):
    fact_key: str = Field(..., description="Fact state key required.")
    expected_value: Any = Field(..., description="Target value required for fact.")
    description: str = Field(default="Required precondition")


class LLMPostconditionItem(BaseModel):
    fact_key: str = Field(..., description="Fact state key modified.")
    target_value: Any = Field(..., description="Target value resulting from action.")
    description: str = Field(default="Target postcondition")


class LLMDependencyItem(BaseModel):
    prerequisite_action_type: Optional[str] = Field(default=None)
    prerequisite_fact_key: Optional[str] = Field(default=None)
    required_value: Optional[Any] = Field(default=None)
    failure_outcome: str = Field(default="BLOCK")
    description: str = Field(default="Dependency requirement")


class ExtractedActionPayload(BaseModel):
    action_type: str = Field(..., description="Identifier of the proposed action.")
    target_entity_id: str = Field(..., description="Target entity ID or identifier.")
    risk_level: str = Field(default="LOW", description="Risk level: LOW, MEDIUM, HIGH, CRITICAL.")
    preconditions: List[LLMPreconditionItem] = Field(default_factory=list)
    postconditions: List[LLMPostconditionItem] = Field(default_factory=list)
    dependencies: List[LLMDependencyItem] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


ACTION_COMPILATION_SCHEMA = ExtractedActionPayload.model_json_schema()


class LLMActionCompiler(BaseActionCompiler):
    """LLM-backed Action Compiler relying on BaseLLMProvider for structured semantic compilation."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        system_prompt: str = ACTION_COMPILATION_SYSTEM_PROMPT,
    ) -> None:
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt

    def compile_proposal(self, proposal: AgentProposal) -> Action:
        """Translates an AgentProposal into a strongly-typed Action primitive via BaseLLMProvider."""
        logger.debug("LLMActionCompiler processing proposal id=%s from agent=%s", proposal.id, proposal.agent_name)

        user_prompt = (
            f"Agent: {proposal.agent_name}\n"
            f"Proposed Intent: {proposal.proposed_intent}\n"
            f"Reasoning: {proposal.natural_language_reasoning}\n"
            f"Raw Payload: {json.dumps(proposal.raw_action_payload)}"
        )

        request = LLMRequest(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            structured_schema=ACTION_COMPILATION_SCHEMA,
            temperature=0.0,
        )

        try:
            response = self.llm_provider.generate(request)
        except LLMError as e:
            logger.error("LLM provider error during semantic action compilation: %s", str(e))
            raise SemanticCompilerError(f"LLM semantic compilation failed: {str(e)}") from e

        # Validate & parse structured output
        raw_output: Any = None
        if response.content:
            try:
                raw_output = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse LLM action compilation content as JSON: %s", response.content)
                raise SemanticCompilerError(f"Invalid JSON response from LLM compiler: {str(e)}") from e
        elif response.structured_output is not None:
            raw_output = response.structured_output

        if not isinstance(raw_output, dict):
            raise SemanticCompilerError("LLM action compiler response did not return a valid dictionary.")

        try:
            payload = ExtractedActionPayload.model_validate(raw_output)
        except ValidationError as e:
            logger.error("Pydantic validation error for LLM action payload: %s", str(e))
            raise SemanticCompilerError(f"Failed LLM payload schema validation: {str(e)}") from e

        # Validate RiskLevel
        risk_str = payload.risk_level.upper()
        try:
            risk_level = ActionRiskLevel(risk_str)
        except ValueError:
            raise SemanticCompilerError(
                f"Invalid risk level '{payload.risk_level}'. Must be one of {[r.value for r in ActionRiskLevel]}."
            )

        # Convert Preconditions
        preconditions: List[Precondition] = [
            Precondition(
                fact_key=p.fact_key,
                expected_value=p.expected_value,
                description=p.description,
            )
            for p in payload.preconditions
        ]

        # Convert Postconditions
        postconditions: List[Postcondition] = [
            Postcondition(
                fact_key=p.fact_key,
                target_value=p.target_value,
                description=p.description,
            )
            for p in payload.postconditions
        ]

        # Convert Dependencies
        dependencies: List[Dependency] = []
        for d in payload.dependencies:
            try:
                failure_outcome = ValidationVerdict(d.failure_outcome.upper())
            except ValueError:
                failure_outcome = ValidationVerdict.BLOCK

            dependencies.append(
                Dependency(
                    prerequisite_action_type=d.prerequisite_action_type,
                    prerequisite_fact_key=d.prerequisite_fact_key,
                    required_value=d.required_value,
                    failure_outcome=failure_outcome,
                    description=d.description,
                )
            )

        # Build Action preserving proposal_id traceability
        compiled_action = Action(
            proposal_id=proposal.id,
            agent_name=proposal.agent_name,
            action_type=payload.action_type,
            target_entity_id=payload.target_entity_id,
            risk_level=risk_level,
            preconditions=preconditions,
            postconditions=postconditions,
            dependencies=dependencies,
            parameters=payload.parameters,
        )

        logger.info(
            "LLMActionCompiler successfully compiled Action id=%s for proposal_id=%s.",
            compiled_action.id,
            proposal.id,
        )
        return compiled_action
