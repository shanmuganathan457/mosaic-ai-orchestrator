"""MOSAIC Deterministic Mock Domain Agents.

These agents simulate specialized domain workers (Security, Billing, Subscription, Access).
They are deterministic, offline components designed for research and pipeline testing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from uuid import UUID

from mosaic.domain.models.schemas import AgentProposal, SubTask, IntentSpan


class BaseMockAgent(ABC):
    """Abstract base class for deterministic domain mock agents."""

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Name of the specialized domain agent."""
        pass

    @abstractmethod
    def process_subtask(
        self, subtask: SubTask, intent_span: IntentSpan
    ) -> AgentProposal:
        """Processes an assigned subtask and generates a candidate AgentProposal.

        Args:
            subtask: Child task assigned to this agent.
            intent_span: Extracted intent span containing verbatim evidence.

        Returns:
            Unvalidated AgentProposal object.
        """
        pass


class SecurityMockAgent(BaseMockAgent):
    """Deterministic Mock Agent for Account Security & Identity tasks."""

    @property
    def agent_name(self) -> str:
        return "SecurityMockAgent"

    def process_subtask(
        self, subtask: SubTask, intent_span: IntentSpan
    ) -> AgentProposal:
        return AgentProposal(
            sub_task_id=subtask.id,
            agent_name=self.agent_name,
            proposed_intent=intent_span.intent_name,
            natural_language_reasoning=f"Detected account security concern in '{intent_span.verbatim_text}'. Recommending account restriction pending identity verification.",
            raw_action_payload={
                "action_type": "restrict_account",
                "target_entity": intent_span.metadata.get("account_id", "user_account_default"),
                "risk_level": "HIGH",
                "preconditions": [
                    {"fact_key": "fraud_alert_active", "expected_value": True, "description": "Security flag must be active"}
                ],
                "postconditions": [
                    {"fact_key": "account_status", "target_value": "restricted", "description": "Mark account as restricted"}
                ]
            }
        )


class BillingMockAgent(BaseMockAgent):
    """Deterministic Mock Agent for Billing & Payment tasks."""

    @property
    def agent_name(self) -> str:
        return "BillingMockAgent"

    def process_subtask(
        self, subtask: SubTask, intent_span: IntentSpan
    ) -> AgentProposal:
        payment_id = intent_span.metadata.get("payment_id", "payment_default")
        return AgentProposal(
            sub_task_id=subtask.id,
            agent_name=self.agent_name,
            proposed_intent=intent_span.intent_name,
            natural_language_reasoning=f"Customer requested refund for transaction '{payment_id}'. Proposing refund execution.",
            raw_action_payload={
                "action_type": "refund_payment",
                "target_entity": payment_id,
                "risk_level": "HIGH",
                "preconditions": [
                    {"fact_key": "payment_verified", "expected_value": True, "description": "Payment record must be verified"},
                    {"fact_key": "identity_verified", "expected_value": True, "description": "Customer identity must be verified for financial transactions"}
                ],
                "postconditions": [
                    {"fact_key": "refund_status", "target_value": "processed", "description": "Set refund status to processed"}
                ]
            }
        )


class SubscriptionMockAgent(BaseMockAgent):
    """Deterministic Mock Agent for Subscription & Plan tasks."""

    @property
    def agent_name(self) -> str:
        return "SubscriptionMockAgent"

    def process_subtask(
        self, subtask: SubTask, intent_span: IntentSpan
    ) -> AgentProposal:
        subscription_id = intent_span.metadata.get("subscription_id", "sub_default")
        return AgentProposal(
            sub_task_id=subtask.id,
            agent_name=self.agent_name,
            proposed_intent=intent_span.intent_name,
            natural_language_reasoning=f"Customer requested cancellation for subscription '{subscription_id}'. Proposing subscription cancellation.",
            raw_action_payload={
                "action_type": "cancel_subscription",
                "target_entity": subscription_id,
                "risk_level": "MEDIUM",
                "preconditions": [
                    {"fact_key": "subscription_active", "expected_value": True, "description": "Subscription must currently be active"}
                ],
                "postconditions": [
                    {"fact_key": "subscription_status", "target_value": "cancelled", "description": "Set subscription status to cancelled"}
                ]
            }
        )


class AccessMockAgent(BaseMockAgent):
    """Deterministic Mock Agent for Login & Account Access tasks."""

    @property
    def agent_name(self) -> str:
        return "AccessMockAgent"

    def process_subtask(
        self, subtask: SubTask, intent_span: IntentSpan
    ) -> AgentProposal:
        account_id = intent_span.metadata.get("account_id", "user_account_default")
        return AgentProposal(
            sub_task_id=subtask.id,
            agent_name=self.agent_name,
            proposed_intent=intent_span.intent_name,
            natural_language_reasoning=f"Customer requested access restoration for account '{account_id}'. Proposing login access restoration.",
            raw_action_payload={
                "action_type": "restore_login_access",
                "target_entity": account_id,
                "risk_level": "MEDIUM",
                "preconditions": [
                    {"fact_key": "identity_verified", "expected_value": True, "description": "Identity must be verified before restoring access"}
                ],
                "postconditions": [
                    {"fact_key": "account_status", "target_value": "active", "description": "Set account status to active"}
                ],
                "dependencies": [
                    {
                        "prerequisite_action_type": "verify_identity",
                        "failure_outcome": "BLOCK",
                        "description": "Identity verification action must complete first"
                    }
                ]
            }
        )
