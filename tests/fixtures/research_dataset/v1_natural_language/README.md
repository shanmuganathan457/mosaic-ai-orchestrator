# MOSAIC Research Benchmark Dataset v1 (Natural-Language Variants)

This directory contains natural-language variants of the authoritative `v1/cases.json` benchmark dataset.

---

## 1. Dataset Purpose & Methodology

The `v1_natural_language` dataset evaluates natural-language robustness across LLM intent decomposition and semantic action compilation strategies while maintaining identical ground truth verdicts and conflict definitions.

Each benchmark case in this dataset:
- Maps to an authoritative source benchmark case in `v1/cases.json` via `source_case_id`.
- Contains realistic linguistic variations including conversational wording, reordered intents, emphatic requests, and polite inquiries.
- Preserves explicit ground truth (`expected_intents`, `expected_agent_actions`, `expected_final_verdict`, `expected_conflicts`, `initial_facts`, `active_flags`).

---

## 2. Dataset Structure & Statistics

- **Total Natural-Language Variant Cases:** 36 cases.
- **Variants per Selected Case:** 3 natural-language variants per source case.
- **Authoritative Ground-Truth Parity:** 100% aligned with `v1/cases.json`.

---

## 3. Ground Truth Preservation Rule

> [!IMPORTANT]
> Ground truth verdicts (`ALLOW`, `BLOCK`, `ESCALATED`), expected intents, and conflict types are fixed and independent of LLM model output. Model outputs are evaluated against this dataset without dynamic adaptation or post-hoc ground truth alteration.
