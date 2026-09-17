"""Unit tests for MOSAIC domain models and validation interfaces."""

from uuid import uuid4
from fastapi.testclient import TestClient

from mosaic.domain.models import (
    Action,
    ActionRiskLevel,
    AgentProposal,
    Case,
    CaseState,
    CaseStatus,
    Conflict,
    ConflictType,
    IntentCategory,
    IntentSpan,
    Postcondition,
    Precondition,
    ValidationResult,
    ValidationVerdict,
)
from mosaic.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 and expected payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "MOSAIC" in data["project"]


def test_domain_model_instantiation():
    """Verify domain models instantiate correctly with strict validation."""
    case_id = uuid4()
    
    # Create CaseState
    state = CaseState(
        case_id=case_id,
        current_facts={"identity_verified": False, "payment_verified": True},
        active_flags=["ACCOUNT_RESTRICTED"]
    )
    
    # Create IntentSpan
    intent = IntentSpan(
        category=IntentCategory.BILLING,
        intent_name="REFUND_REQUEST",
        confidence=0.95,
        verbatim_text="I want a refund for today's charge",
        start_char=45,
        end_char=79
    )
    
    # Create Case
    case = Case(
        id=case_id,
        customer_id="cust_12345",
        raw_message="My account was hacked and I want a refund",
        status=CaseStatus.NEW,
        intents=[intent],
        state=state
    )
    
    assert case.id == case_id
    assert case.intents[0].category == IntentCategory.BILLING
    assert case.state.current_facts["identity_verified"] is False


def test_structured_action_preconditions():
    """Verify compilation of structured Action with Preconditions and Postconditions."""
    proposal_id = uuid4()
    
    action = Action(
        proposal_id=proposal_id,
        agent_name="BillingAgent",
        action_type="REFUND_PAYMENT",
        target_entity_id="payment_99812",
        risk_level=ActionRiskLevel.HIGH,
        preconditions=[
            Precondition(
                fact_key="payment_verified",
                expected_value=True,
                description="Payment must be confirmed in payment gateway"
            ),
            Precondition(
                fact_key="identity_verified",
                expected_value=True,
                description="Customer identity must be verified prior to refund"
            )
        ],
        postconditions=[
            Postcondition(
                fact_key="refund_status",
                target_value="PENDING",
                description="Refund queued for settlement"
            )
        ]
    )
    
    assert action.agent_name == "BillingAgent"
    assert len(action.preconditions) == 2
    assert action.preconditions[0].fact_key == "payment_verified"
    assert action.risk_level == ActionRiskLevel.HIGH


def test_validation_result_block_construction():
    """Verify construction of ValidationResult with conflict diagnostic payload."""
    case_id = uuid4()
    action_id = uuid4()
    
    conflict = Conflict(
        conflict_type=ConflictType.PRECONDITION_UNSATISFIED,
        action_id=action_id,
        reason_code="PRECONDITION_FAILED",
        message="Precondition 'identity_verified=True' failed. Current fact value is False.",
        affected_fact_key="identity_verified"
    )
    
    val_result = ValidationResult(
        case_id=case_id,
        verdict=ValidationVerdict.BLOCK,
        blocked_action_ids=[action_id],
        conflicts=[conflict]
    )
    
    assert val_result.verdict == ValidationVerdict.BLOCK
    assert val_result.conflicts[0].conflict_type == ConflictType.PRECONDITION_UNSATISFIED
    assert val_result.conflicts[0].affected_fact_key == "identity_verified"
