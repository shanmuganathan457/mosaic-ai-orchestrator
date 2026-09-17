"""MOSAIC End-to-End Pipeline Unit Tests.

Executes controlled test dataset (Case 1 through Case 10) through:
Raw Controlled Case -> Mock Agents -> Agent Proposals -> Semantic State Compiler -> Validation Engine -> Expected Verdict.
"""

from typing import Dict, Type
import pytest

from mosaic.agents import (
    AccessMockAgent,
    BaseMockAgent,
    BillingMockAgent,
    SecurityMockAgent,
    SubscriptionMockAgent,
)
from mosaic.compiler import SemanticStateCompiler
from mosaic.domain.models import SubTask, ValidationVerdict
from mosaic.validation.engine import DefaultValidationEngine
from tests.fixtures.controlled_dataset import get_controlled_test_cases

AGENT_MAP: Dict[str, BaseMockAgent] = {
    "Security": SecurityMockAgent(),
    "Billing": BillingMockAgent(),
    "Subscription": SubscriptionMockAgent(),
    "Access": AccessMockAgent(),
}


@pytest.fixture
def compiler():
    return SemanticStateCompiler()


@pytest.fixture
def validator():
    return DefaultValidationEngine()


@pytest.mark.parametrize("case_data", get_controlled_test_cases(), ids=lambda c: c["name"])
def test_end_to_end_controlled_cases(case_data, compiler, validator):
    """Executes each case in the 10-case controlled dataset through the end-to-end pipeline."""
    case_id = case_data["case_id"]
    state = case_data["initial_state"]
    intents = case_data["intents"]
    policies = case_data["policies"]
    expected_verdict = case_data["expected_verdict"]

    agent_keys = case_data.get("assigned_agent_keys", [case_data.get("assigned_agent_key")])
    
    compiled_actions = []

    # 1. Process via Mock Domain Agents -> Proposals -> Compiler -> Actions
    for i, intent in enumerate(intents):
        agent_key = agent_keys[i]
        agent = AGENT_MAP[agent_key]
        subtask = SubTask(case_id=case_id, intent_span_id=intent.id, assigned_agent=agent.agent_name)
        
        # Mock Agent produces AgentProposal
        proposal = agent.process_subtask(subtask, intent)
        
        # Semantic State Compiler compiles Proposal into Action
        action = compiler.compile_proposal(proposal)
        compiled_actions.append(action)

    # 2. Evaluate via Validation Engine
    result = validator.validate(case_state=state, proposed_actions=compiled_actions, policies=policies)

    # 3. Assert Expected Verdict
    assert result.verdict == expected_verdict, (
        f"Failed for '{case_data['name']}': Expected verdict {expected_verdict}, got {result.verdict}. "
        f"Conflicts: {result.conflicts}"
    )


def test_end_to_end_state_dependent_demonstration(compiler, validator):
    """Explicit E2E test verifying exact SAME action under State A (unverified) -> BLOCK, State B (verified) -> ALLOW."""
    billing_agent = BillingMockAgent()
    
    # Define single intent span
    intent = get_controlled_test_cases()[8]["intents"][0]
    
    # 1. Generate Proposal
    subtask = SubTask(case_id=intent.id, intent_span_id=intent.id, assigned_agent=billing_agent.agent_name)
    proposal = billing_agent.process_subtask(subtask, intent)
    
    # 2. Compile Action
    action = compiler.compile_proposal(proposal)
    
    # 3. Evaluate under State 9 (Unverified)
    state_unverified = get_controlled_test_cases()[8]["initial_state"]
    result_unverified = validator.validate(case_state=state_unverified, proposed_actions=[action], policies=[])
    assert result_unverified.verdict == ValidationVerdict.BLOCK
    assert len(result_unverified.blocked_action_ids) == 1

    # 4. Evaluate exact SAME compiled action under State 10 (Verified)
    state_verified = get_controlled_test_cases()[9]["initial_state"]
    result_verified = validator.validate(case_state=state_verified, proposed_actions=[action], policies=[])
    assert result_verified.verdict == ValidationVerdict.ALLOW
    assert len(result_verified.allowed_action_ids) == 1
