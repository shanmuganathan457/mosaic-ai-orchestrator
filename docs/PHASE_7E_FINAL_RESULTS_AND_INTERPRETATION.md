# MOSAIC Phase 7E Final Results and Research Interpretation

## 1. Executive Summary

Phase 7E evaluated the MOSAIC multi-agent governance architecture against Baseline A (`baseline_a_single_intent`) and Baseline B (`baseline_b_direct_multi_agent`) under **Design C (shared-context evaluation)** using a real LLM provider (**Ollama**, `llama3.2:latest`). The dataset comprised 39 natural language customer support cases (`v1_natural_language`).

The benchmark evaluated two primary dimensions:
1. **End-to-End Governance Execution**: Evaluating whether deterministic validation correctly blocks invalid action proposals, permits valid ones, and escalates ambiguous requests.
2. **Hierarchical Conflict Conformance**: Evaluating the structural alignment between MOSAIC's deterministic validation engine (which distinguishes *primary root conflicts* from *secondary downstream cascades*) and ground-truth conflict annotations.

Out of 39 attempted benchmark cases, **36 completed successfully** and **3 encountered upstream LLM/API errors** (2 evidence-hallucination errors during intake and 1 Ollama request timeout during action compilation). On the 36 completed cases, MOSAIC achieved **100% verdict accuracy** (36/36), correctly blocking all invalid requests and escalating all ambiguous ones. 

To resolve incomplete `expected_conflicts` annotations in the historical v1 dataset, an offline rescore was conducted against a versioned ground-truth dataset (**Phase 7E conflict-gold-v2**). Against Gold-v2, MOSAIC achieved **100% primary conflict conformance** (26/26 TP, 0 FP, 0 FN) and **100% secondary conflict conformance** (21/21 TP, 0 FP, 0 FN).

**Critical Research Interpretation Note**: The Gold-v2 offline rescore results demonstrate **conformance between MOSAIC's validation output and the finalized Phase 7E Gold-v2 annotation scheme**. It is **NOT** an independent external validation of general real-world conflict-detection accuracy, as Gold-v2 annotations were explicitly refined to formalize the hierarchical validation semantics established in Phase 7E.

---

## 2. Experiment Setup

- **LLM Provider**: Ollama (`http://localhost:11434`)
- **Model Identifier**: `llama3.2:latest` (3.2B parameter instruction-tuned model)
- **Dataset**: `tests/fixtures/research_dataset/v1_natural_language/cases.json` (39 natural language test cases across 13 base case families with 3 prompt variants each)
- **Evaluation Design**: Design C / Shared-Context Evaluation. A shared context builder decomposes user intents and compiles actions once per case, passing identical semantic artifacts to Baseline A, Baseline B, and MOSAIC to isolate governance logic from LLM generation variance.
- **Git Commit Provenance**:
  - Implementation HEAD: `1979573` (`fix(phase-7e): preserve conflict hierarchy in evaluation results`)
  - Methodology HEAD: `79c07c6` (`docs(phase-7e): document benchmark methodology and interpretation`)
- **Prompt Leakage & Execution Integrity**: No ground-truth annotations were supplied in any LLM prompt. Deterministic validation operated strictly on compiled state transitions. All offline rescore operations executed with 0 LLM/API network calls.

---

## 3. Observed Real-LLM Benchmark Results

The table below presents the empirical results observed during the real-LLM benchmark execution stored in `evaluation_results/phase_7e_design_c_v2/results.json`. All metrics are reported conditional on the **36 completed cases**.

| Metric | Baseline A | Baseline B | MOSAIC Validated |
| :--- | :---: | :---: | :---: |
| **Completed Cases** | 36 / 39 | 36 / 39 | 36 / 39 |
| **Verdict Accuracy** | 27.78% (10/36) | 27.78% (10/36) | **100.00% (36/36)** |
| **Intent Precision** | 100.00% | 100.00% | 100.00% |
| **Intent Recall** | 76.60% | 100.00% | 100.00% |
| **Intent F1 Score** | 86.75% | 100.00% | 100.00% |
| **Action Coverage** | 76.60% | 100.00% | 100.00% |
| **Unexpected Action Rate** | 0.00% | 0.00% | 0.00% |
| **Escalation Precision** | 100.00% | 100.00% | 100.00% |
| **False Escalation Rate** | 0.00% | 0.00% | 0.00% |
| **Validation Latency (Mean)** | 0.1 ms | 0.0 ms | 76.8 ms |
| **Validation Latency (P95)** | 0.1 ms | 0.1 ms | 2.5 ms |
| **Total Prompt Tokens** | 35,417 | 35,417 | 35,417 |
| **Total Completion Tokens** | 11,459 | 11,459 | 11,459 |
| **Total Tokens Consumed** | 46,876 | 46,876 | 46,876 |

### Failed Case Breakdown (3/39 Cases)
1. **`case_001_variant_01`**: `LLMResponseError` — LLM returned hallucinated `verbatim_text 'refund, charge pay_101'` not traceable to raw customer message during intake.
2. **`case_002_variant_01`**: `LLMResponseError` — LLM returned hallucinated `verbatim_text 'refund payment'` not traceable to raw customer message during intake.
3. **`case_005_variant_02`**: `SemanticCompilerError` — Local Ollama HTTP request timed out after 60.0s during action compilation.

*Note: In accordance with Design C shared-context rules, when an upstream LLM failure occurs during intake or compilation, all three evaluation systems fail for that case, preserving strict comparative parity.*

---

## 4. Hierarchical Validation Semantics

Phase 7E introduced formal hierarchical validation semantics within MOSAIC's deterministic validation engine (`src/mosaic/validation/engine.py`).

### Severity & Verdict Precedence
Validation evaluates candidate action proposals against security, compliance, and dependency policies, applying strict verdict precedence:
$$\text{ESCALATED} > \text{BLOCK} > \text{ALLOW}$$

1. **`ESCALATED`**: Triggers when policy flags (e.g., `SUSPICIOUS_LOCATION_LOGIN`, `PENDING_LEGAL_HOLD`, `MANUAL_AUDIT_REQUIRED`) or ambiguous intent spans are present. Takes top priority to prevent automated processing of high-risk cases.
2. **`BLOCK`**: Triggers when explicit rule violations (missing dependencies, unsatisfied preconditions, or mutually exclusive postconditions) are detected.
3. **`ALLOW`**: Issued only when all proposed actions satisfy state preconditions, dependency prerequisites, and safety policies.

### Conflict Hierarchy: Primary vs. Secondary Conflicts
When validation blocks or escalates execution, the engine distinguishes between root cause conflicts and downstream cascading failures:

- **Primary Conflicts**: The root semantic conflict that directly determines the routing decision.
  - `AMBIGUOUS_EVIDENCE`: Unresolvable slot ambiguities or security flags (triggers `ESCALATED`).
  - `MISSING_DEPENDENCY`: Unmet prerequisite actions (e.g., attempting `restore_login_access` without completing `verify_identity`). This represents the primary root block.
- **Secondary Conflicts**: Downstream structural or state consequences caused by a primary block.
  - `PRECONDITION_UNSATISFIED`: Precondition checks that fail as a direct consequence of a missing prerequisite action (e.g., `identity_verified == False` when identity verification was not performed).
  - Short-circuit rule: Actions that are blocked by missing dependencies are excluded from cross-action postcondition collision checks (`MUTUALLY_EXCLUSIVE_POSTCONDITION`).

---

## 5. Gold-v2 Annotation Revision

The original historical dataset (`v1/cases.json`) contained single `expected_conflicts` labels that were inconsistent with hierarchical validation semantics. For example:
- **Cases 003 & 009** (conflicting account lock vs. unlock requests): Originally labeled as `MUTUALLY_EXCLUSIVE_POSTCONDITION`. However, because `restore_login_access` requires `verify_identity` (which was not present in initial facts), the dependency check fails *before* postcondition checks execute. Thus, the primary root conflict is `MISSING_DEPENDENCY`.
- **Case 004** (locked account access restore without verified identity): Originally labeled as `PRECONDITION_UNSATISFIED`. The primary root conflict is `MISSING_DEPENDENCY` (`verify_identity`), while `PRECONDITION_UNSATISFIED` (`identity_verified == False`) is a secondary downstream consequence.
- **Cases 021–023** (security policy escalation cases): Originally labeled solely with `AMBIGUOUS_EVIDENCE`. Because unverified identity parameters accompany the ambiguous access restore request, secondary `MISSING_DEPENDENCY` and `PRECONDITION_UNSATISFIED` conflicts are also generated.

To formalize these structural relationships without mutating historical benchmark baselines, a versioned annotation file was created: `tests/fixtures/research_dataset/v1_natural_language_phase7e_gold_v2/cases.json`.

---

## 6. Offline Gold-v2 Rescore

An offline rescore script (`tools/rescore_phase_7e_conflict_gold_v2.py`) evaluated MOSAIC's 36 completed prediction records from `phase_7e_design_c_v2/results.json` against `Phase 7E conflict-gold-v2`.

### Summary of Conflict Rescore Metrics (36 Completed Cases)

| Metric Layer | Gold Target Field | Prediction Field | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Conflicts** | `expected_conflicts` | `primary_conflicts` | 26 | 0 | 0 | **100.00%** | **100.00%** | **100.00%** |
| **Secondary Conflicts** | `expected_secondary_conflicts` | `secondary_conflicts` | 21 | 0 | 0 | **100.00%** | **100.00%** | **100.00%** |
| **Flat Conflicts** *(Supplementary)* | `expected_conflicts` | `detected_conflicts` | 26 | 21 | 0 | 55.32% | **100.00%** | 71.23% |

### Key Findings & Interpretation:
1. **Primary Conformance**: MOSAIC achieved 100% Precision, Recall, and F1 on primary root conflict detection.
2. **Secondary Conformance**: MOSAIC achieved 100% Precision, Recall, and F1 on secondary downstream conflict detection.
3. **Flat Metric Interpretation**: The flat conflict metric yields a lower Precision (55.32%) and F1 (71.23%) because it aggregates secondary cascading conflicts (`MISSING_DEPENDENCY`, `PRECONDITION_UNSATISFIED`) against a single primary gold label. The 21 "flat false positives" are structurally valid secondary detections, not false alarms.

---

## 7. Per-Conflict-Type Results

The table below details per-conflict-type performance across Primary, Secondary, and Flat evaluations. Zero-support categories where $\text{TP} = \text{FP} = \text{FN} = 0$ are explicitly reported as **N/A**.

| Conflict Type | Layer | Gold Count | TP | FP | FN | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AMBIGUOUS_EVIDENCE`** | Primary | 9 | 9 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| | Secondary | 0 | 0 | 0 | 0 | N/A | N/A | **N/A** |
| | Flat | 9 | 9 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| **`MISSING_DEPENDENCY`** | Primary | 12 | 12 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| | Secondary | 9 | 9 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| | Flat | 12 | 12 | 9 | 0 | 57.14% | 100.00% | **72.73%** |
| **`PRECONDITION_UNSATISFIED`** | Primary | 5 | 5 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| | Secondary | 12 | 12 | 0 | 0 | 100.00% | 100.00% | **100.00%** |
| | Flat | 5 | 5 | 12 | 0 | 29.41% | 100.00% | **45.45%** |
| **`MUTUALLY_EXCLUSIVE_POSTCONDITION`** | Primary | 0 | 0 | 0 | 0 | N/A | N/A | **N/A** |
| | Secondary | 0 | 0 | 0 | 0 | N/A | N/A | **N/A** |
| | Flat | 0 | 0 | 0 | 0 | N/A | N/A | **N/A** |

---

## 8. What Phase 7E Demonstrates

### Supported Findings:
- **Deterministic Governance Validation & Specification Conformance**: Deterministic policy validation effectively prevents unauthorized or unvalidated execution, achieving 100% verdict accuracy on completed real-LLM outputs and demonstrating 100% internal conformance with the finalized Phase 7E Gold-v2 specification.
- **Operational Parity via Design C**: Shared-context evaluation successfully isolates governance and rule validation from LLM generation variance.
- **Internal Consistency**: MOSAIC's validation engine exhibits structural consistency with the Gold-v2 hierarchical conflict specification.
- **Fault Tolerance**: The resumable benchmark architecture survives LLM timeouts and restarts without data loss or corruption.

### Unsupported Claims (Explicit Non-Goals):
- **Generalization**: Does not establish conflict detection performance on unseen, unmodeled, or out-of-domain enterprise workflows.
- **Statistical Significance**: The 39-case dataset is not sized for formal statistical hypothesis testing or power analysis.
- **External Accuracy**: 100% Gold-v2 conformance is an internal specification alignment proof, not an independent external validation of general real-world conflict-detection accuracy.

---

## 9. Limitations

1. **Dataset Scope**: Small synthetic dataset (39 cases across 13 core case families).
2. **Upstream LLM Failures**: 3 out of 39 cases failed due to intake/compilation errors, limiting empirical sample size to 36 completed cases.
3. **Gold Revision Alignment**: Gold-v2 was derived after inspecting validation engine mechanics. Conformance metrics reflect specification alignment rather than double-blind evaluation.
4. **Latency Tail**: Mean validation latency of 76.8 ms includes a single 2,729.91 ms outlier; P95 latency was 2.5 ms.

---

## 10. Research Claim Boundaries

### Claims We Can Make:
- "MOSAIC's deterministic validation engine achieved 100% verdict accuracy across 36 completed real-LLM benchmark cases under Design C evaluation."
- "MOSAIC's primary and secondary conflict classifications exhibit 100% structural conformance with the Phase 7E Gold-v2 annotation schema."
- "Design C shared-context evaluation enables reproducible, head-to-head comparative analysis of multi-agent governance logic."

### Claims We Should NOT Make:
- "MOSAIC is proven to have 100% conflict detection accuracy in arbitrary real-world environments."
- "MOSAIC is statistically proven superior to Baseline A or Baseline B."
- "The Gold-v2 rescore represents an independent, double-blind external benchmark validation."

---

## 11. Reproducibility & Artifact Directory

All experimental data, scripts, and documentation are version-controlled and immutable:

- **Historical Benchmark Results**: `evaluation_results/phase_7e_design_c_v2/results.json`
- **Conflict Gold-v2 Dataset**: `tests/fixtures/research_dataset/v1_natural_language_phase7e_gold_v2/cases.json`
- **Offline Rescore Artifacts**: `evaluation_results/phase_7e_design_c_v2_conflict_gold_v2/conflict_rescore.json`
- **Offline Rescore Script**: [rescore_phase_7e_conflict_gold_v2.py](file:///c:/Users/janas/Documents/GitHub/mosaic-ai-orchestrator/tools/rescore_phase_7e_conflict_gold_v2.py)
- **Methodology Specification**: `docs/INFRASTRUCTURE_NOTE_PHASE_7E.md`
- **Implementation Commit SHA**: `1979573`
- **Documentation Commit SHA**: `79c07c6`

---

## 12. Phase 7E Status

Phase 7E evaluation, validation, and research interpretation are **COMPLETE and FROZEN**. All code, test suites, datasets, and benchmark reports are verified and ready for review.
