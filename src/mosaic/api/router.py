"""MOSAIC Support Intake API Endpoint Router."""

import logging
from fastapi import APIRouter, HTTPException, status

from mosaic.api.policies import DEFAULT_API_POLICIES
from mosaic.api.schemas import SupportRequest
from mosaic.executor import GovernanceExecutionResponse, MockBackendExecutor
from mosaic.orchestrator import MosaicOrchestrator

logger = logging.getLogger("mosaic.api.router")

router = APIRouter(prefix="/api/v1/support", tags=["Support Intake & Governance"])

# Singleton instances for default API operations
default_orchestrator = MosaicOrchestrator()
default_executor = MockBackendExecutor()


@router.post(
    "/requests",
    response_model=GovernanceExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Customer Support Request for Governance & Execution",
    description="""
Processes a customer support request through the end-to-end MOSAIC pipeline:
1. Intake & Intent Decomposition
2. Agent Proposal Generation
3. Semantic Action Compilation
4. Deterministic State & Policy Validation (ALLOW / BLOCK / ESCALATE)
5. Backend Execution via Mock Executor (ONLY allowed actions reach executor)
""",
)
async def process_support_request(
    payload: SupportRequest,
) -> GovernanceExecutionResponse:
    """Processes customer support request and returns complete governance lifecycle response."""
    logger.info("API request received for customer_id=%s", payload.customer_id)

    if not payload.raw_message or not payload.raw_message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="raw_message cannot be empty",
        )

    try:
        # Step 1: Intake -> Decomposition -> Compilation -> Validation
        case, subtask_mappings = default_orchestrator.intake_engine.create_case_from_intake(
            customer_id=payload.customer_id,
            raw_message=payload.raw_message,
            initial_facts=payload.initial_facts,
            active_flags=payload.active_flags,
        )

        compiled_actions = []
        for subtask, intent_span, agent_name in subtask_mappings:
            agent = default_orchestrator.agent_registry.get(agent_name)
            if agent:
                proposal = agent.process_subtask(subtask, intent_span)
                action = default_orchestrator.compiler.compile_proposal(proposal)
                compiled_actions.append(action)

        validation_result = default_orchestrator.validation_engine.validate(
            case_state=case.state,
            proposed_actions=compiled_actions,
            policies=DEFAULT_API_POLICIES,
        )

        # Step 2: Governance Execution via Mock Backend Executor
        response = default_executor.execute_governed_case(
            customer_id=payload.customer_id,
            raw_message=payload.raw_message,
            detected_intents=case.intents,
            proposed_actions=compiled_actions,
            current_state=case.state.current_facts,
            validation_result=validation_result,
        )

        return response

    except Exception as e:
        logger.exception("Error processing support request: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failure: {str(e)}",
        )
