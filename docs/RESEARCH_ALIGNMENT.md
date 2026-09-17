# MOSAIC Research Alignment Matrix

This document maps software components in the **MOSAIC** codebase to the research concepts, requirements, and hypotheses specified in `docs/MOSAIC_Research_Document.docx`.

---

## 1. Research Objectives & System Mapping

| Research Concept / Requirement | Software Component / Module | Implementation Purpose |
| :--- | :--- | :--- |
| **Multi-Intent & Evidence Extraction** | `mosaic.intake.decomposer.IntentDecompositionEngine` | Rule-based, deterministic intent extractor converting raw text into evidence-backed `IntentSpan` objects. |
| **Case & Sub-Task Tracking** | `mosaic.domain.models.schemas` (`Case`, `SubTask`, `CaseState`) | Maintains state records, current facts, active flags, and completed action graphs for multi-intent customer communications. |
| **Agent Action Proposals** | `mosaic.agents.mock_agents` | Deterministic mock worker agents generating `AgentProposal` objects. |
| **Semantic Action Compilation** | `mosaic.compiler.state_compiler.SemanticStateCompiler` | Translates natural language proposals into structured, machine-checkable `Action` primitives with explicit state requirements. |
| **State & Policy Validation** | `mosaic.validation.engine.DefaultValidationEngine` | Evaluates structured actions against case facts, preconditions, postconditions, and business constraints deterministically without calling an LLM. |
| **End-to-End Pipeline Coordination** | `mosaic.orchestrator.MosaicOrchestrator` | Coordinates raw text intake, mock agent execution, state compilation, and validation evaluation. |
| **Explainable Verdicts & Conflict Reporting** | `mosaic.domain.models.schemas` (`ValidationResult`, `Conflict`) | Produces machine-readable and human-readable diagnostic conflict reports (`reason_code`, `message`, `failed_conditions`). |
| **Human Escalation Control** | `mosaic.domain.models.schemas` (`Escalation`) | Formulates structured escalation payloads with human-readable conflict diagnostics when actions are blocked or ambiguous. |
| **Response Synthesis Guard** | `mosaic.domain.models.schemas` (`ResponseDraft`) | Ensures final responses are generated strictly from validated (`ALLOW`) outcomes. |

---

## 2. Research Alignment Details

### 2.1 State-Dependent Validation (Not Static Keyword Rules)
Static rules like `("account_locked", "refund") -> BLOCK` are insufficient for enterprise workflows. MOSAIC models state dynamically:
- If `CaseState.current_facts["identity_verified"] == False`, a `refund_payment` action is `BLOCK`ed due to missing preconditions.
- If `CaseState.current_facts["identity_verified"] == True`, the exact same `refund_payment` action evaluates to `ALLOW`.

### 2.2 Modular Validation Flow
Validation is decoupled into four explicit stages:
1. **Precondition Evaluation:** Check required pre-state key/value facts.
2. **Dependency Evaluation:** Verify required prerequisite actions/facts.
3. **Policy Constraint Evaluation:** Check declarative `PolicyRule` objects against active flags and forbidden state conditions.
4. **Cross-Action Conflict Detection:** Detect mutually exclusive postconditions between candidate actions.

---

## 3. Implementation Status Summary

### Currently Implemented
- [x] Pydantic Domain Schemas (`Case`, `CaseState`, `Action`, `Precondition`, `Postcondition`, `Dependency`, `PolicyRule`, `ValidationResult`, `Conflict`)
- [x] Sub-Evaluators (`PreconditionEvaluator`, `DependencyEvaluator`, `PolicyEvaluator`, `PostconditionConflictEvaluator`)
- [x] Deterministic Orchestration Engine (`DefaultValidationEngine`)
- [x] Semantic State Compiler & Deterministic Mock Domain Agents (`mosaic.compiler`, `mosaic.agents`)
- [x] Deterministic Intent & Evidence Decomposition Engine (`mosaic.intake`)
- [x] End-to-End Pipeline Orchestrator (`MosaicOrchestrator`)
- [x] 29 passing pytest unit & end-to-end pipeline tests (`tests/`)

### Intentionally NOT Yet Implemented
- [ ] Provider-Agnostic LLM Interface (Phase 5)
- [ ] Frontend Dashboard / UI
- [ ] Real CRM / Payment Gateway Integrations
