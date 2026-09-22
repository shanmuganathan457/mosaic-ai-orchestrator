"""MOSAIC Default API Governance Policies.

This module defines the default set of PolicyRule instances that are
applied at the API layer when processing customer support requests.

These policies translate active case flags (supplied by the caller)
into deterministic validation outcomes via the PolicyEvaluator.
They do NOT duplicate the dependency/precondition logic embedded in
mock agent payloads — they add flag-aware escalation/blocking rules
that are specific to the support API runtime context.
"""

from mosaic.domain.models import PolicyRule, ValidationVerdict


# ---------------------------------------------------------------------------
# Security & Access Policies
# ---------------------------------------------------------------------------

POLICY_SUSPICIOUS_LOGIN_ESCALATE = PolicyRule(
    id="pol_access_suspicious_location_escalate",
    rule_name="Suspicious Location Login: Escalate Access Restoration",
    description=(
        "When the active case flag SUSPICIOUS_LOCATION_LOGIN is present, "
        "any restore_login_access action must be escalated to a human agent "
        "for manual review rather than automatically allowed or blocked."
    ),
    condition_type="MANDATORY_FLAG_CHECK",
    action_types=["restore_login_access"],
    forbidden_active_flags=["SUSPICIOUS_LOCATION_LOGIN"],
    outcome=ValidationVerdict.ESCALATED,
    explanation=(
        "SUSPICIOUS_LOCATION_LOGIN flag indicates the login attempt originated "
        "from an unexpected geographic location. Automated access restoration is "
        "not permitted; a human agent must verify account ownership."
    ),
    is_active=True,
)


# ---------------------------------------------------------------------------
# Default policy set applied to all API support requests
# ---------------------------------------------------------------------------

DEFAULT_API_POLICIES: list[PolicyRule] = [
    POLICY_SUSPICIOUS_LOGIN_ESCALATE,
]
