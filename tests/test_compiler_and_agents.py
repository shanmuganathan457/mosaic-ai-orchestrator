"""Unit tests for Mock Agents and Semantic State Compiler."""

from uuid import uuid4
import pytest

from mosaic.agents import (
    AccessMockAgent,
    BillingMockAgent,
    SecurityMockAgent,
    SubscriptionMockAgent,
)
from mosaic.compiler import SemanticCompilerError, SemanticStateCompiler
from mosaic.domain.models import (
    ActionRiskLevel,
    AgentProposal,
    IntentCategory,
    IntentSpan,
    SubTask,
)


def test_mock_agents_proposal_generation():
    """Verify mock domain agents generate valid AgentProposal objects."""
    case_id = uuid4()
    intent_span = IntentSpan(
        category=IntentCategory.BILLING,
        intent_name="refund_payment",
        confidence=0.95,
        verbatim_text="Refund my money",
        start_char=0,
        end_char=15,
        metadata={"payment_id": "pay_555"}
    )
    subtask = SubTask(case_id=case_id, intent_span_id=intent_span.id, assigned_agent="BillingMockAgent")

    billing_agent = BillingMockAgent()
    proposal = billing_agent.process_subtask(subtask, intent_span)

    assert isinstance(proposal, AgentProposal)
    assert proposal.agent_name == "BillingMockAgent"
    assert proposal.raw_action_payload["action_type"] == "refund_payment"
    assert proposal.raw_action_payload["target_entity"] == "pay_555"


def test_semantic_state_compiler_successful_compilation():
    """Verify SemanticStateCompiler translates AgentProposal into structured Action."""
    compiler = SemanticStateCompiler()
    subtask_id = uuid4()
    
    proposal = AgentProposal(
        sub_task_id=subtask_id,
        agent_name="SecurityMockAgent",
        proposed_intent="restrict_account",
        natural_language_reasoning="Fraud flag set",
        raw_action_payload={
            "action_type": "restrict_account",
            "target_entity": "acc_123",
            "risk_level": "HIGH",
            "preconditions": [
                {"fact_key": "fraud_alert_active", "expected_value": True, "description": "Security flag active"}
            ],
            "postconditions": [
                {"fact_key": "account_status", "target_value": "restricted", "description": "Set status restricted"}
            ]
        }
    )

    action = compiler.compile_proposal(proposal)

    assert action.agent_name == "SecurityMockAgent"
    assert action.action_type == "restrict_account"
    assert action.target_entity_id == "acc_123"
    assert action.risk_level == ActionRiskLevel.HIGH
    assert len(action.preconditions) == 1
    assert action.preconditions[0].fact_key == "fraud_alert_active"
    assert len(action.postconditions) == 1
    assert action.postconditions[0].target_value == "restricted"


def test_semantic_state_compiler_missing_action_type():
    """Verify compiler raises SemanticCompilerError when payload is missing action_type."""
    compiler = SemanticStateCompiler()
    proposal = AgentProposal(
        sub_task_id=uuid4(),
        agent_name="FaultyAgent",
        proposed_intent="unknown",
        natural_language_reasoning="No action type provided",
        raw_action_payload={"risk_level": "LOW"}
    )

    with pytest.raises(SemanticCompilerError):
        compiler.compile_proposal(proposal)
