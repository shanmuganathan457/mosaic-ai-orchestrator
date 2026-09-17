"""Tests for LLM-based Semantic Action Compiler (Phase 5C)."""

import pytest
from uuid import uuid4

from mosaic.compiler import (
    BaseActionCompiler,
    LLMActionCompiler,
    SemanticCompilerError,
    SemanticStateCompiler,
)
from mosaic.domain.models import (
    Action,
    ActionRiskLevel,
    AgentProposal,
    CaseState,
    IntentSpan,
    Postcondition,
    Precondition,
    SubTask,
    ValidationVerdict,
)
from mosaic.llm import LLMResponse, MockLLMProvider
from mosaic.orchestrator import MosaicOrchestrator
from mosaic.validation import DefaultValidationEngine


def test_llm_action_compiler_valid_refund_proposal():
    """Test LLMActionCompiler compiling a valid refund proposal into a domain Action."""
    valid_json = """{
        "action_type": "refund_payment",
        "target_entity_id": "pay_101",
        "risk_level": "HIGH",
        "preconditions": [
            {"fact_key": "payment_verified", "expected_value": true}
        ],
        "postconditions": [
            {"fact_key": "refund_status", "target_value": "processed"}
        ],
        "dependencies": []
    }"""
    mock_llm = MockLLMProvider(default_response_text=valid_json)
    compiler = LLMActionCompiler(llm_provider=mock_llm)

    proposal = AgentProposal(
        sub_task_id=uuid4(),
        agent_name="BillingMockAgent",
        proposed_intent="refund_payment",
        natural_language_reasoning="Customer requested refund for pay_101",
        raw_action_payload={"action_type": "refund_payment", "target_entity": "pay_101"},
    )

    action = compiler.compile_proposal(proposal)

    assert isinstance(action, Action)
    assert action.action_type == "refund_payment"
    assert action.target_entity_id == "pay_101"
    assert action.risk_level == ActionRiskLevel.HIGH
    assert action.proposal_id == proposal.id
    assert len(action.preconditions) == 1
    assert action.preconditions[0].fact_key == "payment_verified"
    assert action.preconditions[0].expected_value is True
    assert len(action.postconditions) == 1
    assert action.postconditions[0].fact_key == "refund_status"
    assert action.postconditions[0].target_value == "processed"


def test_llm_action_compiler_traceability_preserved():
    """Verify that the Action retains proposal_id from AgentProposal."""
    response_json = """{
        "action_type": "cancel_subscription",
        "target_entity_id": "sub_404",
        "risk_level": "MEDIUM",
        "preconditions": [],
        "postconditions": [],
        "dependencies": []
    }"""
    mock_llm = MockLLMProvider(default_response_text=response_json)
    compiler = LLMActionCompiler(llm_provider=mock_llm)

    proposal = AgentProposal(
        sub_task_id=uuid4(),
        agent_name="SubscriptionMockAgent",
        proposed_intent="cancel_subscription",
        natural_language_reasoning="User wants to cancel sub_404",
        raw_action_payload={"action_type": "cancel_subscription", "target_entity": "sub_404"},
    )

    action = compiler.compile_proposal(proposal)
    assert action.proposal_id == proposal.id


def test_llm_action_compiler_malformed_json_raises_error():
    """Test that malformed LLM response JSON raises SemanticCompilerError."""
    mock_llm = MockLLMProvider(default_response_text="INVALID JSON {{{")
    compiler = LLMActionCompiler(llm_provider=mock_llm)

    proposal = AgentProposal(
        sub_task_id=uuid4(),
        agent_name="BillingMockAgent",
        proposed_intent="refund_payment",
        natural_language_reasoning="Refund attempt",
        raw_action_payload={"action_type": "refund_payment"},
    )

    with pytest.raises(SemanticCompilerError, match="Invalid JSON response"):
        compiler.compile_proposal(proposal)


def test_llm_action_compiler_missing_required_field_raises_error():
    """Test that missing required field in LLM response raises SemanticCompilerError."""
    missing_field_json = """{
        "target_entity_id": "pay_101",
        "risk_level": "HIGH"
    }"""
    mock_llm = MockLLMProvider(default_response_text=missing_field_json)
    compiler = LLMActionCompiler(llm_provider=mock_llm)

    proposal = AgentProposal(
        sub_task_id=uuid4(),
        agent_name="BillingMockAgent",
        proposed_intent="refund_payment",
        natural_language_reasoning="Refund attempt",
        raw_action_payload={"action_type": "refund_payment"},
    )

    with pytest.raises(SemanticCompilerError, match="Failed LLM payload schema validation"):
        compiler.compile_proposal(proposal)


def test_llm_action_compiler_invalid_risk_level_raises_error():
    """Test that an unsupported risk level in LLM response raises SemanticCompilerError."""
    invalid_risk_json = """{
        "action_type": "refund_payment",
        "target_entity_id": "pay_101",
        "risk_level": "EXTREME_DANGER",
        "preconditions": [],
        "postconditions": [],
        "dependencies": []
    }"""
    mock_llm = MockLLMProvider(default_response_text=invalid_risk_json)
    compiler = LLMActionCompiler(llm_provider=mock_llm)

    proposal = AgentProposal(
        sub_task_id=uuid4(),
        agent_name="BillingMockAgent",
        proposed_intent="refund_payment",
        natural_language_reasoning="Refund attempt",
        raw_action_payload={"action_type": "refund_payment"},
    )

    with pytest.raises(SemanticCompilerError, match="Invalid risk level"):
        compiler.compile_proposal(proposal)


def test_end_to_end_state_dependent_validation():
    """End-to-End Test: State A (payment_verified=false -> BLOCK) vs State B (payment_verified=true -> ALLOW).
    
    Verifies that the LLM compiler produces the exact same structured action, and ONLY the ValidationEngine
    determines the ALLOW/BLOCK verdict based on state.
    """
    valid_json = """{
        "action_type": "refund_payment",
        "target_entity_id": "pay_101",
        "risk_level": "HIGH",
        "preconditions": [
            {"fact_key": "payment_verified", "expected_value": true},
            {"fact_key": "identity_verified", "expected_value": true}
        ],
        "postconditions": [
            {"fact_key": "refund_status", "target_value": "processed"}
        ],
        "dependencies": []
    }"""
    mock_llm = MockLLMProvider(default_response_text=valid_json)
    compiler = LLMActionCompiler(llm_provider=mock_llm)
    orchestrator = MosaicOrchestrator(compiler=compiler)

    # State A: payment_verified = false, identity_verified = true -> BLOCK
    result_state_a = orchestrator.process_customer_case(
        customer_id="cust_001",
        raw_message="Please refund my payment pay_101",
        initial_facts={"payment_verified": False, "identity_verified": True},
    )

    # State B: payment_verified = true, identity_verified = true -> ALLOW
    result_state_b = orchestrator.process_customer_case(
        customer_id="cust_001",
        raw_message="Please refund my payment pay_101",
        initial_facts={"payment_verified": True, "identity_verified": True},
    )

    # The compiler did not decide the verdict; the deterministic ValidationEngine did.
    assert result_state_a.verdict == ValidationVerdict.BLOCK
    assert result_state_b.verdict == ValidationVerdict.ALLOW


def test_end_to_end_cross_action_conflict():
    """End-to-End Test: Two compiled actions produce incompatible postconditions (account_status=restricted vs active).
    
    The LLM semantic compiler creates the Actions; the ValidationEngine detects the conflict and returns BLOCK.
    """
    class MultiResponseMockLLMProvider(MockLLMProvider):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def generate(self, prompt, system_prompt: str = None) -> LLMResponse:
            self.call_count += 1
            if self.call_count == 1:
                json_data = """{
                    "action_type": "restrict_account",
                    "target_entity_id": "acc_999",
                    "risk_level": "HIGH",
                    "preconditions": [],
                    "postconditions": [
                        {"fact_key": "account_status", "target_value": "restricted"}
                    ],
                    "dependencies": []
                }"""
            else:
                json_data = """{
                    "action_type": "restore_login_access",
                    "target_entity_id": "acc_999",
                    "risk_level": "MEDIUM",
                    "preconditions": [],
                    "postconditions": [
                        {"fact_key": "account_status", "target_value": "active"}
                    ],
                    "dependencies": []
                }"""
            return LLMResponse(
                provider_name="MockLLMProvider",
                model_name="mock-model",
                content=json_data,
                raw_response={"mock": True}
            )

    multi_mock_llm = MultiResponseMockLLMProvider()
    compiler = LLMActionCompiler(llm_provider=multi_mock_llm)
    orchestrator = MosaicOrchestrator(compiler=compiler)

    # Multi-intent customer message triggering SecurityMockAgent ("hacked") + AccessMockAgent ("restore access")
    raw_message = "My account acc_999 was hacked, please restore access right now."
    result = orchestrator.process_customer_case(
        customer_id="cust_002",
        raw_message=raw_message,
        initial_facts={"account_status": "active", "identity_verified": True, "fraud_alert_active": True},
    )

    # Validation engine should detect postcondition conflict (restricted vs active) -> BLOCK
    assert result.verdict == ValidationVerdict.BLOCK
    assert len(result.conflicts) > 0
    assert any("conflicts with Action" in conflict.message for conflict in result.conflicts)


def test_deterministic_compiler_baseline_unaffected():
    """Verify that SemanticStateCompiler continues to inherit from BaseActionCompiler and work baseline."""
    deterministic_compiler = SemanticStateCompiler()
    assert isinstance(deterministic_compiler, BaseActionCompiler)

    proposal = AgentProposal(
        sub_task_id=uuid4(),
        agent_name="BillingMockAgent",
        proposed_intent="refund_payment",
        natural_language_reasoning="Test baseline",
        raw_action_payload={"action_type": "refund_payment", "target_entity": "pay_101"},
    )

    action = deterministic_compiler.compile_proposal(proposal)
    assert isinstance(action, Action)
    assert action.action_type == "refund_payment"
