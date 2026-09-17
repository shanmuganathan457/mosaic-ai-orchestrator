"""MOSAIC Controlled Test Dataset.

This module provides a deterministic dataset of 10 controlled research test cases,
covering single-intent, multi-intent, dependency, missing prerequisite, cross-action conflict,
ambiguous state, and state-dependent scenarios.
"""

from typing import Any, Dict, List
from uuid import UUID, uuid4

from mosaic.domain.models.schemas import (
    Case,
    CaseState,
    CaseStatus,
    IntentCategory,
    IntentSpan,
    PolicyRule,
    SubTask,
    ValidationVerdict,
)


def get_controlled_test_cases() -> List[Dict[str, Any]]:
    """Returns list of 10 deterministic test cases with raw text, states, intents, and expected outcomes."""
    
    # -------------------------------------------------------------
    # CASE 1: Single-Intent Valid Case -> ALLOW
    # -------------------------------------------------------------
    case1_id = uuid4()
    case1 = {
        "case_id": case1_id,
        "name": "Case 1: Single-Intent Valid Subscription Cancellation",
        "raw_message": "Please cancel my recurring pro subscription.",
        "initial_state": CaseState(
            case_id=case1_id,
            current_facts={"subscription_active": True}
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.SUBSCRIPTION,
                intent_name="cancel_subscription",
                confidence=0.98,
                verbatim_text="cancel my recurring pro subscription",
                start_char=7,
                end_char=44,
                metadata={"subscription_id": "sub_101"}
            )
        ],
        "assigned_agent_key": "Subscription",
        "policies": [],
        "expected_verdict": ValidationVerdict.ALLOW,
    }

    # -------------------------------------------------------------
    # CASE 2: Single-Intent Invalid Case (Missing Prerequisite) -> BLOCK
    # -------------------------------------------------------------
    case2_id = uuid4()
    case2 = {
        "case_id": case2_id,
        "name": "Case 2: Refund Request with Unverified Payment",
        "raw_message": "I want a refund for my charge.",
        "initial_state": CaseState(
            case_id=case2_id,
            current_facts={"payment_verified": False, "identity_verified": True}
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.BILLING,
                intent_name="refund_payment",
                confidence=0.95,
                verbatim_text="I want a refund for my charge",
                start_char=0,
                end_char=29,
                metadata={"payment_id": "pay_202"}
            )
        ],
        "assigned_agent_key": "Billing",
        "policies": [],
        "expected_verdict": ValidationVerdict.BLOCK,
    }

    # -------------------------------------------------------------
    # CASE 3: Two Independent Valid Intents -> ALLOW
    # -------------------------------------------------------------
    case3_id = uuid4()
    case3 = {
        "case_id": case3_id,
        "name": "Case 3: Independent Subscription Cancel + Refund",
        "raw_message": "Cancel my subscription and refund my payment.",
        "initial_state": CaseState(
            case_id=case3_id,
            current_facts={"subscription_active": True, "payment_verified": True, "identity_verified": True}
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.SUBSCRIPTION,
                intent_name="cancel_subscription",
                confidence=0.96,
                verbatim_text="Cancel my subscription",
                start_char=0,
                end_char=22,
                metadata={"subscription_id": "sub_303"}
            ),
            IntentSpan(
                category=IntentCategory.BILLING,
                intent_name="refund_payment",
                confidence=0.95,
                verbatim_text="refund my payment",
                start_char=27,
                end_char=44,
                metadata={"payment_id": "pay_303"}
            )
        ],
        "assigned_agent_keys": ["Subscription", "Billing"],
        "policies": [],
        "expected_verdict": ValidationVerdict.ALLOW,
    }

    # -------------------------------------------------------------
    # CASE 4: Multi-Intent Case with Completed Action Dependency -> ALLOW
    # -------------------------------------------------------------
    case4_id = uuid4()
    case4 = {
        "case_id": case4_id,
        "name": "Case 4: Access Restoration with Completed Identity Verification",
        "raw_message": "I cannot log in, please restore my access.",
        "initial_state": CaseState(
            case_id=case4_id,
            current_facts={"identity_verified": True},
            completed_action_types=["verify_identity"]
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.ACCESS_RESTORATION,
                intent_name="restore_login_access",
                confidence=0.97,
                verbatim_text="restore my access",
                start_char=24,
                end_char=41,
                metadata={"account_id": "acc_404"}
            )
        ],
        "assigned_agent_key": "Access",
        "policies": [],
        "expected_verdict": ValidationVerdict.ALLOW,
    }

    # -------------------------------------------------------------
    # CASE 5: Missing Prerequisite Dependency -> BLOCK
    # -------------------------------------------------------------
    case5_id = uuid4()
    case5 = {
        "case_id": case5_id,
        "name": "Case 5: Access Restoration without Completed Identity Verification",
        "raw_message": "Restore my login access immediately.",
        "initial_state": CaseState(
            case_id=case5_id,
            current_facts={"identity_verified": True},
            completed_action_types=[]  # verify_identity NOT completed
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.ACCESS_RESTORATION,
                intent_name="restore_login_access",
                confidence=0.97,
                verbatim_text="Restore my login access",
                start_char=0,
                end_char=23,
                metadata={"account_id": "acc_505"}
            )
        ],
        "assigned_agent_key": "Access",
        "policies": [],
        "expected_verdict": ValidationVerdict.BLOCK,
    }

    # -------------------------------------------------------------
    # CASE 6: Cross-Action Postcondition Conflict -> BLOCK
    # -------------------------------------------------------------
    case6_id = uuid4()
    case6 = {
        "case_id": case6_id,
        "name": "Case 6: Security Lock vs Login Restoration Conflict",
        "raw_message": "My account was compromised but I need to log in now.",
        "initial_state": CaseState(
            case_id=case6_id,
            current_facts={"identity_verified": True, "fraud_alert_active": True},
            completed_action_types=["verify_identity"]
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.SECURITY,
                intent_name="restrict_account",
                confidence=0.94,
                verbatim_text="account was compromised",
                start_char=3,
                end_char=25,
                metadata={"account_id": "acc_606"}
            ),
            IntentSpan(
                category=IntentCategory.ACCESS_RESTORATION,
                intent_name="restore_login_access",
                confidence=0.93,
                verbatim_text="need to log in now",
                start_char=32,
                end_char=50,
                metadata={"account_id": "acc_606"}
            )
        ],
        "assigned_agent_keys": ["Security", "Access"],
        "policies": [],
        "expected_verdict": ValidationVerdict.BLOCK,
    }

    # -------------------------------------------------------------
    # CASE 7: Ambiguous / Missing State -> ESCALATE
    # -------------------------------------------------------------
    case7_id = uuid4()
    case7 = {
        "case_id": case7_id,
        "name": "Case 7: High Risk Action with Ambiguous State",
        "raw_message": "Restore access for suspicious account.",
        "initial_state": CaseState(
            case_id=case7_id,
            current_facts={},  # Missing identity_verified fact key
            completed_action_types=["verify_identity"]
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.ACCESS_RESTORATION,
                intent_name="restore_login_access",
                confidence=0.91,
                verbatim_text="Restore access",
                start_char=0,
                end_char=14,
                metadata={"account_id": "acc_707"}
            )
        ],
        "assigned_agent_key": "Access",
        "policies": [],
        "expected_verdict": ValidationVerdict.BLOCK,  # Missing identity_verified precondition yields BLOCK
    }

    # -------------------------------------------------------------
    # CASE 8: Policy Violation (Active Flag FORBIDDEN) -> BLOCK
    # -------------------------------------------------------------
    case8_id = uuid4()
    case8 = {
        "case_id": case8_id,
        "name": "Case 8: Refund Attempt under Active Fraud Restrict Flag",
        "raw_message": "Process my refund.",
        "initial_state": CaseState(
            case_id=case8_id,
            current_facts={"payment_verified": True, "identity_verified": True},
            active_flags=["ACCOUNT_RESTRICTED"]
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.BILLING,
                intent_name="refund_payment",
                confidence=0.96,
                verbatim_text="Process my refund",
                start_char=0,
                end_char=17,
                metadata={"payment_id": "pay_808"}
            )
        ],
        "assigned_agent_key": "Billing",
        "policies": [
            PolicyRule(
                id="POL_NO_FINANCIAL_ON_RESTRICTED",
                rule_name="No Financial Disbursements on Restricted Accounts",
                description="Financial transactions prohibited if account has ACCOUNT_RESTRICTED flag",
                action_types=["refund_payment"],
                forbidden_active_flags=["ACCOUNT_RESTRICTED"],
                explanation="Cannot refund payments while account restriction flag is active."
            )
        ],
        "expected_verdict": ValidationVerdict.BLOCK,
    }

    # -------------------------------------------------------------
    # CASE 9: State-Dependent Validation (Unverified -> BLOCK)
    # -------------------------------------------------------------
    case9_id = uuid4()
    case9 = {
        "case_id": case9_id,
        "name": "Case 9: State-Dependent Refund (Unverified Identity State)",
        "raw_message": "Refund payment 909.",
        "initial_state": CaseState(
            case_id=case9_id,
            current_facts={"payment_verified": True, "identity_verified": False}
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.BILLING,
                intent_name="refund_payment",
                confidence=0.99,
                verbatim_text="Refund payment 909",
                start_char=0,
                end_char=18,
                metadata={"payment_id": "pay_909"}
            )
        ],
        "assigned_agent_key": "Billing",
        "policies": [],
        "expected_verdict": ValidationVerdict.BLOCK,
    }

    # -------------------------------------------------------------
    # CASE 10: State-Dependent Validation (Verified -> ALLOW)
    # -------------------------------------------------------------
    case10_id = uuid4()
    case10 = {
        "case_id": case10_id,
        "name": "Case 10: State-Dependent Refund (Verified Identity State)",
        "raw_message": "Refund payment 909.",
        "initial_state": CaseState(
            case_id=case10_id,
            current_facts={"payment_verified": True, "identity_verified": True}
        ),
        "intents": [
            IntentSpan(
                category=IntentCategory.BILLING,
                intent_name="refund_payment",
                confidence=0.99,
                verbatim_text="Refund payment 909",
                start_char=0,
                end_char=18,
                metadata={"payment_id": "pay_909"}
            )
        ],
        "assigned_agent_key": "Billing",
        "policies": [],
        "expected_verdict": ValidationVerdict.ALLOW,
    }

    return [case1, case2, case3, case4, case5, case6, case7, case8, case9, case10]
