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

- **Total Natural-Language Variant Cases:** 39 cases.
- **Selected Source Cases:** 13 source cases.
- **Variants per Selected Case:** 3 natural-language variants per source case.
- **Authoritative Ground-Truth Parity:** 100% aligned with `v1/cases.json`.

The dataset cardinality is distinct from a particular benchmark run. Phase 7E
Design C v2 attempted all 39 dataset cases; 36 completed and 3 failed upstream
LLM processing. Those completed/failed counts describe that observed run, not
the dataset size.

---

## 3. Ground Truth Preservation Rule

> [!IMPORTANT]
> Ground truth verdicts (`ALLOW`, `BLOCK`, `ESCALATED`), expected intents, and conflict types are fixed and independent of LLM model output. Model outputs are evaluated against this dataset without dynamic adaptation or post-hoc ground truth alteration.

## 4. Phase 7E Conflict Annotation Guidance

`expected_conflicts` currently represents this benchmark's conflict annotation.
Finalized Phase 7E semantics distinguish primary/root conflicts from
secondary/downstream conflicts. A future versioned gold dataset may introduce
`expected_secondary_conflicts` to represent the latter explicitly.

Cases 003, 004, and 009 require annotation review before conflict metrics are
used as headline research results. The original `v1_natural_language` dataset,
including `cases.json`, must remain unchanged; any approved correction must use
a new versioned dataset.
