# MOSAIC Research Benchmark Dataset v1

This directory contains version 1 (`v1`) of the synthetic research benchmark dataset for evaluating cross-agent semantic state validation.

---

## 1. Dataset Purpose & Methodology

The MOSAIC benchmark dataset evaluates multi-intent customer-support automation across three architectural systems:
1. **Baseline A:** Single-intent routing without cross-action validation.
2. **Baseline B:** Direct multi-intent synthesis without state validation.
3. **MOSAIC:** Multi-intent decomposition with state-aware deterministic validation.

---

## 2. Category & Verdict Distribution

The corrected dataset contains **23 total benchmark cases** with the following distribution:

- **10 ALLOW cases:** Multi-intent and single-intent operations where all preconditions, dependencies, and state transitions are satisfied without conflict.
- **10 BLOCK cases:** Operations with failing preconditions, missing dependencies, or conflicting state mutations.
- **3 ESCALATED cases:** Scenarios representing operational ambiguity or unresolved compliance/security flags where safe execution cannot be deterministically guaranteed.

### Benchmark Categories (8 distinct scenario types):

1. **`independent_valid`**: Multi-intent inquiries where all proposed actions are valid and compatible. Expected verdict: `ALLOW`.
2. **`missing_precondition`**: Actions requiring state facts (`payment_verified`, `identity_verified`) that are missing or false. Expected verdict: `BLOCK`.
3. **`failed_dependency`**: Actions requiring prerequisite actions (e.g., identity verification) that have not been completed. Expected verdict: `BLOCK`.
4. **`cross_action_conflict`**: Multiple agent actions proposing incompatible postcondition state changes (e.g., `account_status = restricted` vs `active`). Expected verdict: `BLOCK`.
5. **`state_dependent_valid`**: Scenarios testing state-dependent execution bounds. Expected verdict: `ALLOW`.
6. **`state_dependent_invalid`**: Scenarios testing state-dependent execution bounds when facts fail. Expected verdict: `BLOCK`.
7. **`insufficient_info_escalate`**: Inquiries with ambiguous evidence (e.g., suspicious login location, active legal hold, mandatory audit flags) requiring human intervention. Expected verdict: `ESCALATED`.
8. **`no_conflict_multi_intent`**: Multi-intent inquiries touching separate domains (e.g., refund + subscription cancellation) without state collision. Expected verdict: `ALLOW`.

---

## 3. Escalation Ground-Truth Definition

A case is designated as **`ESCALATED`** when:
- Required security/identity verification cannot be established from case context (e.g., `SUSPICIOUS_LOCATION_LOGIN`).
- Active legal or compliance restrictions freeze account modifications (e.g., `PENDING_LEGAL_HOLD`).
- Critical state information is ambiguous or conflicting evidence cannot be safely reconciled (e.g., `MANUAL_AUDIT_REQUIRED`).

Escalation is **NOT** triggered merely because multiple actions exist or by naive string keyword matching. Each `ESCALATED` case explicitly tests whether the system safely halts automated execution when deterministic safety bounds cannot be guaranteed.

---

## 4. Intent Comparison Methodology

Intent evaluation uses set-based micro-aggregated precision, recall, and F1 metrics:

$$\text{True Positives (TP)} = | \text{Predicted Intents} \cap \text{Expected Intents} |$$
$$\text{False Positives (FP)} = | \text{Predicted Intents} \setminus \text{Expected Intents} |$$
$$\text{False Negatives (FN)} = | \text{Expected Intents} \setminus \text{Predicted Intents} |$$

Counts are aggregated across all benchmark cases before calculating micro-averaged Precision, Recall, and F1 to prevent skewed per-case averaging. Zero-denominator edge cases return `1.0` if both predicted and expected sets are empty, and `0.0` if one set is empty while the other is non-empty.

---

## 5. Synthetic Benchmark Limitations

> [!NOTE]
> This synthetic benchmark provides a controlled development baseline for evaluating architectural state-validation mechanisms.
> Results must NOT be interpreted as statistically significant proofs of real-world LLM performance or claims of general AI superiority.
