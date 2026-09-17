# MOSAIC Research Benchmark Dataset v1

This directory contains version 1 (`v1`) of the synthetic research benchmark dataset for evaluating cross-agent semantic state validation.

---

## 1. Dataset Purpose & Methodology

The MOSAIC benchmark dataset evaluates multi-intent customer-support automation across three architectural systems:
1. **Baseline A:** Single-intent routing without cross-action validation.
2. **Baseline B:** Direct multi-intent synthesis without state validation.
3. **MOSAIC:** Multi-intent decomposition with state-aware deterministic validation.

---

## 2. Case Categories

The 20 benchmark cases cover 8 distinct scenario categories across **BILLING**, **SECURITY**, **SUBSCRIPTION**, and **ACCESS_RESTORATION**:

1. **`independent_valid`**: Multi-intent inquiries where all proposed actions are valid and compatible. Expected verdict: `ALLOW`.
2. **`missing_precondition`**: Actions requiring state facts (`payment_verified`, `identity_verified`) that are missing or false. Expected verdict: `BLOCK`.
3. **`failed_dependency`**: Actions requiring prerequisite actions (e.g., identity verification) that have not been completed. Expected verdict: `BLOCK`.
4. **`cross_action_conflict`**: Multiple agent actions proposing incompatible postcondition state changes (e.g. `account_status = restricted` vs `active`). Expected verdict: `BLOCK`.
5. **`state_dependent_valid`**: Scenarios testing state-dependent execution bounds. Expected verdict: `ALLOW`.
6. **`state_dependent_invalid`**: Scenarios testing state-dependent execution bounds when facts fail. Expected verdict: `BLOCK`.
7. **`insufficient_info_escalate`**: Inquiries with ambiguous evidence or unresolvable policy constraints requiring human review. Expected verdict: `ESCALATED`.
8. **`no_conflict_multi_intent`**: Multi-intent inquiries touching separate domains (e.g., refund + subscription cancellation) without state collision. Expected verdict: `ALLOW`.

---

## 3. Ground Truth Methodology

Each case explicitly defines:
- **`case_id`**: Unique string identifier.
- **`customer_message`**: Raw customer email/message.
- **`expected_intents`**: Intent names expected to be decomposed.
- **`expected_agent_actions`**: Agent action types expected to be compiled.
- **`initial_facts`** & **`active_flags`**: Initial case state context.
- **`expected_final_verdict`**: Expected ground truth (`ALLOW`, `BLOCK`, `ESCALATED`).
- **`expected_conflicts`**: Array of conflict types if non-ALLOW.
- **`rationale`**: Human-readable rationale explaining the expected verdict.

---

## 4. How to Add Future Cases

To add new benchmark cases to `v1/cases.json`:
1. Ensure the new case follows the `BenchmarkCase` Pydantic schema in `src/mosaic/evaluation/models.py`.
2. Provide explicit `initial_facts` and `expected_final_verdict`.
3. Verify that `expected_conflicts` correctly list relevant `ConflictType` enum strings.
4. Run `python -m pytest tests/test_evaluation_dataset.py` to validate schema correctness.
