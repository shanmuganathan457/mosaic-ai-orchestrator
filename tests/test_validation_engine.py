"""Unit tests for MOSAIC Validation Engine covering Scenarios A through H."""

from uuid import uuid4
import pytest

from mosaic.domain.models import (
    Action,
    ActionRiskLevel,
    CaseState,
    ConflictType,
    Dependency,
    PolicyRule,
    Postcondition,
    Precondition,
    ValidationVerdict,
)
from mosaic.validation.engine import DefaultValidationEngine


@pytest.fixture
def validator():
    return DefaultValidationEngine()


# ==========================================
# TEST A: Simple valid action → ALLOW
# ==========================================
def test_scenario_a_simple_valid_action(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"identity_verified": True, "payment_verified": True}
    )
    action = Action(
        proposal_id=uuid4(),
        agent_name="BillingAgent",
        action_type="refund_payment",
        target_entity_id="pay_101",
        preconditions=[
            Precondition(fact_key="payment_verified", expected_value=True, description="Payment verified")
        ]
    )

    result = validator.validate(case_state=state, proposed_actions=[action], policies=[])

    assert result.verdict == ValidationVerdict.ALLOW
    assert action.id in result.allowed_action_ids
    assert len(result.blocked_action_ids) == 0
    assert len(result.conflicts) == 0


# ==========================================
# TEST B: Missing prerequisite → BLOCK
# ==========================================
def test_scenario_b_missing_prerequisite(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"payment_verified": False}
    )
    action = Action(
        proposal_id=uuid4(),
        agent_name="BillingAgent",
        action_type="refund_payment",
        target_entity_id="pay_102",
        preconditions=[
            Precondition(fact_key="payment_verified", expected_value=True, description="Payment must be verified")
        ]
    )

    result = validator.validate(case_state=state, proposed_actions=[action], policies=[])

    assert result.verdict == ValidationVerdict.BLOCK
    assert action.id in result.blocked_action_ids
    assert len(result.conflicts) == 1
    c = result.conflicts[0]
    assert c.conflict_type == ConflictType.PRECONDITION_UNSATISFIED
    assert c.reason_code == "PRECONDITION_FAILED"
    assert "payment_verified" in c.affected_fact_key


# ==========================================
# TEST C: Multiple prerequisites where one fails → BLOCK
# ==========================================
def test_scenario_c_multiple_prerequisites_one_fails(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"payment_verified": True, "identity_verified": False}
    )
    action = Action(
        proposal_id=uuid4(),
        agent_name="BillingAgent",
        action_type="refund_payment",
        target_entity_id="pay_103",
        preconditions=[
            Precondition(fact_key="payment_verified", expected_value=True, description="Payment verified"),
            Precondition(fact_key="identity_verified", expected_value=True, description="Identity verified"),
        ]
    )

    result = validator.validate(case_state=state, proposed_actions=[action], policies=[])

    assert result.verdict == ValidationVerdict.BLOCK
    assert action.id in result.blocked_action_ids
    assert len(result.conflicts) == 1
    assert result.conflicts[0].affected_fact_key == "identity_verified"


# ==========================================
# TEST D: Two independently valid actions with no conflict → ALLOW
# ==========================================
def test_scenario_d_two_valid_actions_no_conflict(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"subscription_active": True, "payment_verified": True}
    )
    action_1 = Action(
        proposal_id=uuid4(),
        agent_name="SubscriptionAgent",
        action_type="cancel_subscription",
        target_entity_id="sub_99",
        preconditions=[Precondition(fact_key="subscription_active", expected_value=True, description="Active sub")]
    )
    action_2 = Action(
        proposal_id=uuid4(),
        agent_name="BillingAgent",
        action_type="refund_payment",
        target_entity_id="pay_104",
        preconditions=[Precondition(fact_key="payment_verified", expected_value=True, description="Payment verified")]
    )

    result = validator.validate(case_state=state, proposed_actions=[action_1, action_2], policies=[])

    assert result.verdict == ValidationVerdict.ALLOW
    assert len(result.allowed_action_ids) == 2
    assert action_1.id in result.allowed_action_ids
    assert action_2.id in result.allowed_action_ids


# ==========================================
# TEST E: Two actions whose combined postconditions violate a policy / conflict → BLOCK
# ==========================================
def test_scenario_e_conflicting_postconditions(validator):
    case_id = uuid4()
    state = CaseState(case_id=case_id, current_facts={})

    # Action 1 sets account_status = "active"
    action_1 = Action(
        proposal_id=uuid4(),
        agent_name="AccessAgent",
        action_type="restore_login_access",
        target_entity_id="user_55",
        postconditions=[Postcondition(fact_key="account_status", target_value="active", description="Restore active login")]
    )
    # Action 2 sets account_status = "restricted"
    action_2 = Action(
        proposal_id=uuid4(),
        agent_name="SecurityAgent",
        action_type="restrict_account",
        target_entity_id="user_55",
        postconditions=[Postcondition(fact_key="account_status", target_value="restricted", description="Lock account for security")]
    )

    result = validator.validate(case_state=state, proposed_actions=[action_1, action_2], policies=[])

    assert result.verdict == ValidationVerdict.BLOCK
    assert action_1.id in result.blocked_action_ids
    assert action_2.id in result.blocked_action_ids
    assert len(result.conflicts) == 1
    c = result.conflicts[0]
    assert c.conflict_type == ConflictType.MUTUALLY_EXCLUSIVE_POSTCONDITION
    assert c.affected_fact_key == "account_status"


# ==========================================
# TEST F: Dependency has not completed → BLOCK or ESCALATE
# ==========================================
def test_scenario_f_uncompleted_dependency_block(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={},
        completed_action_types=[]  # verify_identity not completed
    )
    action = Action(
        proposal_id=uuid4(),
        agent_name="AccountAgent",
        action_type="execute_account_transfer",
        target_entity_id="acc_777",
        dependencies=[
            Dependency(
                prerequisite_action_type="verify_identity",
                failure_outcome=ValidationVerdict.BLOCK,
                description="Must complete identity verification prior to account transfer"
            )
        ]
    )

    result = validator.validate(case_state=state, proposed_actions=[action], policies=[])

    assert result.verdict == ValidationVerdict.BLOCK
    assert action.id in result.blocked_action_ids
    assert result.conflicts[0].conflict_type == ConflictType.MISSING_DEPENDENCY


# ==========================================
# TEST G: Insufficient information to determine safety → ESCALATE
# ==========================================
def test_scenario_g_insufficient_information_escalate(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"fraud_suspicion_score": "UNKNOWN"}
    )
    action = Action(
        proposal_id=uuid4(),
        agent_name="PayoutAgent",
        action_type="issue_high_value_payout",
        target_entity_id="payout_888",
        dependencies=[
            Dependency(
                prerequisite_fact_key="fraud_clearance_confirmed",
                required_value=True,
                failure_outcome=ValidationVerdict.ESCALATED,
                description="Fraud clearance status must be confirmed for high value payouts"
            )
        ]
    )

    result = validator.validate(case_state=state, proposed_actions=[action], policies=[])

    assert result.verdict == ValidationVerdict.ESCALATED
    assert action.id in result.escalated_action_ids
    assert result.conflicts[0].conflict_type == ConflictType.AMBIGUOUS_EVIDENCE


# ==========================================
# TEST H: Same action evaluated under two different states (State-Dependent Validation)
# ==========================================
def test_scenario_h_state_dependent_validation(validator):
    action_proposal_id = uuid4()

    def create_refund_action():
        return Action(
            proposal_id=action_proposal_id,
            agent_name="BillingAgent",
            action_type="refund_payment",
            target_entity_id="pay_999",
            preconditions=[
                Precondition(fact_key="payment_verified", expected_value=True, description="Payment verified"),
                Precondition(fact_key="identity_verified", expected_value=True, description="Identity verified")
            ]
        )

    # State 1: Unverified state
    state_1 = CaseState(
        case_id=uuid4(),
        current_facts={"payment_verified": True, "identity_verified": False}
    )
    result_1 = validator.validate(case_state=state_1, proposed_actions=[create_refund_action()], policies=[])

    # Assert State 1 produces BLOCK
    assert result_1.verdict == ValidationVerdict.BLOCK
    assert len(result_1.blocked_action_ids) == 1
    assert result_1.conflicts[0].affected_fact_key == "identity_verified"

    # State 2: Verified state
    state_2 = CaseState(
        case_id=uuid4(),
        current_facts={"payment_verified": True, "identity_verified": True}
    )
    result_2 = validator.validate(case_state=state_2, proposed_actions=[create_refund_action()], policies=[])

    # Assert State 2 produces ALLOW
    assert result_2.verdict == ValidationVerdict.ALLOW
    assert len(result_2.allowed_action_ids) == 1
    assert len(result_2.conflicts) == 0


# ==========================================
# TEST I: Escalation demotes Block conflicts
# ==========================================
def test_escalation_primary_vs_secondary(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"fraud_suspicion_score": "UNKNOWN", "payment_verified": False}
    )
    action = Action(
        proposal_id=uuid4(),
        agent_name="PayoutAgent",
        action_type="issue_high_value_payout",
        target_entity_id="payout_888",
        preconditions=[
            Precondition(fact_key="payment_verified", expected_value=True, description="Payment must be verified")
        ],
        dependencies=[
            Dependency(
                prerequisite_fact_key="fraud_clearance_confirmed",
                required_value=True,
                failure_outcome=ValidationVerdict.ESCALATED,
                description="Fraud clearance status must be confirmed"
            )
        ]
    )

    result = validator.validate(case_state=state, proposed_actions=[action], policies=[])

    assert result.verdict == ValidationVerdict.ESCALATED
    assert action.id in result.escalated_action_ids
    assert len(result.primary_conflicts) == 1
    assert result.primary_conflicts[0].conflict_type == ConflictType.AMBIGUOUS_EVIDENCE
    assert len(result.secondary_conflicts) == 1
    assert result.secondary_conflicts[0].conflict_type == ConflictType.PRECONDITION_UNSATISFIED


# ==========================================
# TEST J: Dependency Root Cause demotes Precondition
# ==========================================
def test_dependency_root_cause_vs_precondition(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"identity_verified": False},
        completed_action_types=[]
    )
    action = Action(
        proposal_id=uuid4(),
        agent_name="AccessAgent",
        action_type="restore_login_access",
        target_entity_id="user_123",
        preconditions=[
            Precondition(fact_key="identity_verified", expected_value=True, description="Identity must be verified")
        ],
        dependencies=[
            Dependency(
                prerequisite_action_type="verify_identity",
                failure_outcome=ValidationVerdict.BLOCK,
                description="Identity verification action must complete first"
            )
        ]
    )

    result = validator.validate(case_state=state, proposed_actions=[action], policies=[])

    assert result.verdict == ValidationVerdict.BLOCK
    assert len(result.primary_conflicts) == 1
    assert result.primary_conflicts[0].conflict_type == ConflictType.MISSING_DEPENDENCY
    assert len(result.secondary_conflicts) == 1
    assert result.secondary_conflicts[0].conflict_type == ConflictType.PRECONDITION_UNSATISFIED


# ==========================================
# TEST K: Blocked actions excluded from cross-action eval
# ==========================================
def test_cross_action_ignores_blocked(validator):
    case_id = uuid4()
    state = CaseState(
        case_id=case_id,
        current_facts={"identity_verified": False},
        completed_action_types=[]
    )
    # Action 1 is blocked due to missing dependency
    action_1 = Action(
        proposal_id=uuid4(),
        agent_name="AccessAgent",
        action_type="restore_login_access",
        target_entity_id="user_123",
        dependencies=[
            Dependency(
                prerequisite_action_type="verify_identity",
                failure_outcome=ValidationVerdict.BLOCK,
                description="Must complete identity verification"
            )
        ],
        postconditions=[Postcondition(fact_key="account_status", target_value="active", description="Active")]
    )
    # Action 2 is valid
    action_2 = Action(
        proposal_id=uuid4(),
        agent_name="SecurityAgent",
        action_type="restrict_account",
        target_entity_id="user_123",
        postconditions=[Postcondition(fact_key="account_status", target_value="restricted", description="Restrict")]
    )

    result = validator.validate(case_state=state, proposed_actions=[action_1, action_2], policies=[])

    assert result.verdict == ValidationVerdict.BLOCK
    assert action_1.id in result.blocked_action_ids
    assert action_2.id in result.allowed_action_ids
    # Since action 1 is blocked, no cross conflict is detected between action 1 and 2
    assert not any(c.conflict_type == ConflictType.MUTUALLY_EXCLUSIVE_POSTCONDITION for c in result.conflicts)


# ==========================================
# TEST L: Corrected case_010 semantics in dataset
# ==========================================
def test_case_010_dataset_semantics():
    import json
    from pathlib import Path
    cases_path = Path("tests/fixtures/research_dataset/v1_natural_language/cases.json")
    with open(cases_path) as f:
        cases = json.load(f)
    for c in cases:
        if c["case_id"] == "case_010_variant_01":
            assert c["expected_final_verdict"] == "BLOCK"
            assert "MISSING_DEPENDENCY" in c["expected_conflicts"]
            assert "Phase 7E Update" in c["rationale"]
