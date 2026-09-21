# Phase 7E Design C v2: Benchmark Methodology and Interpretation

## Purpose and observed run

Phase 7E Design C v2 evaluates MOSAIC's deterministic validation layer against
single-intent and direct multi-agent baselines after a shared LLM upstream stage.
The observed benchmark run used Git commit `1979573`, Ollama, and
`llama3.2:latest`.

The run attempted 39 `v1_natural_language` cases. It completed 36 cases and
recorded 3 failed cases: a 92.3% completion rate. It recorded 95 successful LLM
calls and 3 failed LLM calls.

## Design C shared-context architecture

For each attempted case, Design C constructs one shared upstream context: one
LLM intent decomposition followed by one LLM action-compilation call per
proposal. Baseline A, Baseline B, and MOSAIC receive those same decomposed
intents and compiled actions. MOSAIC then obtains its verdict from deterministic
validation; the shared context does not contain benchmark ground truth.

## How to interpret the observed results

All reported verdict, intent, and action metrics are conditional on the 36
completed cases. This run makes no full-39-case accuracy claim; attempted-case
completion and failures are reported separately.

The observed shared upstream LLM cost is 46,876 tokens (35,417 prompt and
11,459 completion tokens). Because Design C shares this upstream work, the same
46,876 tokens must not be interpreted as three independent system costs.

Observed MOSAIC latency over the 36 completed cases was 76.79 ms mean and
2.50 ms P95. The mean includes `case_005_variant_01`, whose recorded latency
was 2,729.91 ms. This outlier is disclosed and retained; it is not removed or
excluded from the reported distribution.

## Conflict-annotation scope

Conflict metrics are not headline Phase 7E research results yet. The current
gold annotations are not aligned with the finalized hierarchy and
candidate-action semantics in all relevant cases.

- Cases 003 and 009 currently expect a mutually exclusive postcondition.
  Under finalized semantics, an action blocked by a missing dependency is not a
  candidate for cross-action comparison, so the observed primary conflict is
  `MISSING_DEPENDENCY`.
- Case 004 currently expects a precondition failure. Under finalized hierarchy,
  `MISSING_DEPENDENCY` is the primary/root conflict and
  `PRECONDITION_UNSATISFIED` is secondary/downstream.

The current dataset has no explicit secondary-conflict gold annotation. A future
versioned annotation dataset may add that field and align cases 003, 004, and
009. Until then, conflict results are descriptive observations rather than
headline accuracy claims.

## Failed cases

The three failures were retained as failed records and excluded from completed-
case metrics:

- `case_001_variant_01`: intent decomposition rejected hallucinated evidence
  text, `refund, charge pay_101`, that was not traceable to the customer
  message.
- `case_002_variant_01`: intent decomposition rejected hallucinated evidence
  text, `refund payment`, that was not traceable to the customer message.
- `case_005_variant_02`: semantic action compilation failed after an Ollama
  request timeout of 60 seconds.

## Artifact preservation

This document interprets the observed benchmark only. The historical artifacts
in `evaluation_results/phase_7e_design_c_v2/` remain unchanged, including
`results.json`, `summary.json`, and `run.log`.
