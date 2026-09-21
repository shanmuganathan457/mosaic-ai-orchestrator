# MOSAIC Natural-Language Conflict Gold v2 (Phase 7E)

## Purpose and relationship to v1

This directory is a versioned conflict-annotation revision of
`../v1_natural_language/`. It contains the same 39 cases, messages, facts,
intents, actions, and final verdicts as v1. The original v1 dataset is
immutable and remains the historical benchmark input.

This is an annotation revision, not a new model benchmark. Creating or using
this gold version requires no Ollama rerun: an offline conflict rescore may use
the recorded primary and secondary MOSAIC conflicts in the existing Phase 7E
Design C v2 result file.

## Conflict fields

`expected_conflicts` denotes expected primary/root conflicts under finalized
Phase 7E execution semantics. `expected_secondary_conflicts` denotes expected
secondary/downstream conflicts and is present on every case; it is `[]` where
no secondary conflict is expected.

The hierarchy treats `AMBIGUOUS_EVIDENCE` as primary for escalation and
`MISSING_DEPENDENCY` as primary when an action lacks an unavailable prerequisite.
`PRECONDITION_UNSATISFIED` may be secondary when it follows from that missing
dependency.

Cross-action `MUTUALLY_EXCLUSIVE_POSTCONDITION` evaluation considers only
actions still executable after per-action prerequisite, dependency, and policy
validation. Blocked actions are excluded from that candidate set.

## v1 annotation changes

Only the conflict annotation fields differ from v1:

- `case_003_variant_01` through `_03`: primary changed from
  `MUTUALLY_EXCLUSIVE_POSTCONDITION` to `MISSING_DEPENDENCY`; no secondary
  conflict is expected because the blocked action never reaches cross-action
  comparison.
- `case_004_variant_01` through `_03`: primary changed from
  `PRECONDITION_UNSATISFIED` to `MISSING_DEPENDENCY`; secondary is
  `PRECONDITION_UNSATISFIED`.
- `case_009_variant_01` through `_03`: primary changed from
  `MUTUALLY_EXCLUSIVE_POSTCONDITION` to `MISSING_DEPENDENCY`; no secondary
  conflict is expected.
- `case_021_variant_01` through `_03`, `case_022_variant_01` through `_03`,
  and `case_023_variant_01` through `_03`: `AMBIGUOUS_EVIDENCE` remains the
  primary escalation conflict. `MISSING_DEPENDENCY` and
  `PRECONDITION_UNSATISFIED` are explicitly annotated as secondary conflicts.
  They follow deterministically from the fixed `restore_login_access` action
  requirements and absent identity-verification state; they are not
  LLM-generated gold labels.

Gold annotations are deduplicated by conflict type, regardless of how many
equivalent actions or conflict instances occur in an observed run.

No other case has been changed automatically. Any future discrepancy must be
reviewed against finalized validator semantics and recorded in a new versioned
dataset rather than by editing v1 or this historical annotation revision.
