# MOSAIC Research Alignment Matrix

This document maps software components in the **MOSAIC** codebase to the research concepts, requirements, and hypotheses specified in `docs/MOSAIC_Research_Document.docx`.

---

## 1. Research Objectives & System Mapping

| Research Concept / Requirement | Software Component / Module | Implementation Purpose |
| :--- | :--- | :--- |
| **Provider-Agnostic LLM Abstraction** | `mosaic.llm.base.BaseLLMProvider`, `mosaic.llm.models` | Establishes a generic Pydantic/ABC boundary (`LLMRequest`, `LLMResponse`) decoupling MOSAIC from provider SDKs. |
| **Deterministic Research Mock LLM** | `mosaic.llm.mock.MockLLMProvider` | Provides a 100% offline, reproducible mock provider for research benchmarking without API/GPU costs. |
| **Multi-Intent & Evidence Extraction** | `mosaic.intake.decomposer.IntentDecompositionEngine` | Rule-based, deterministic intent extractor converting raw text into evidence-backed `IntentSpan` objects. |
| **Case & Sub-Task Tracking** | `mosaic.domain.models.schemas` (`Case`, `SubTask`, `CaseState`) | Maintains state records, current facts, active flags, and completed action graphs for multi-intent customer communications. |
| **Agent Action Proposals** | `mosaic.agents.mock_agents` | Deterministic mock worker agents generating `AgentProposal` objects. |
| **Semantic Action Compilation** | `mosaic.compiler.state_compiler.SemanticStateCompiler` | Translates natural language proposals into structured, machine-checkable `Action` primitives with explicit state requirements. |
| **State & Policy Validation** | `mosaic.validation.engine.DefaultValidationEngine` | Evaluates structured actions against case facts, preconditions, postconditions, and business constraints deterministically without calling an LLM. |
| **End-to-End Pipeline Coordination** | `mosaic.orchestrator.MosaicOrchestrator` | Coordinates raw text intake, mock agent execution, state compilation, and validation evaluation. |
| **Explainable Verdicts & Conflict Reporting** | `mosaic.domain.models.schemas` (`ValidationResult`, `Conflict`) | Produces machine-readable and human-readable diagnostic conflict reports (`reason_code`, `message`, `failed_conditions`). |
| **Human Escalation Control** | `mosaic.domain.models.schemas` (`Escalation`) | Formulates structured escalation payloads with human-readable conflict diagnostics when actions are blocked or ambiguous. |

---

## 2. Research Alignment Details

### 2.1 LLM Abstraction vs. Validation Decoupling
Research requirement: The LLM may propose candidate intents or actions, but **must never make the final safety or validation decision**.
- The `mosaic.llm` abstraction lives strictly in the perception/intake layer.
- The `DefaultValidationEngine` in `mosaic.validation` remains 100% deterministic and LLM-independent.

### 2.2 State-Dependent Validation (Not Static Keyword Rules)
Static rules like `("account_locked", "refund") -> BLOCK` are insufficient for enterprise workflows. MOSAIC models state dynamically:
- If `CaseState.current_facts["identity_verified"] == False`, a `refund_payment` action is `BLOCK`ed due to missing preconditions.
- If `CaseState.current_facts["identity_verified"] == True`, the exact same `refund_payment` action evaluates to `ALLOW`.

---

## 3. Implementation Status Summary

### Currently Implemented
- [x] Pydantic Domain Schemas (`Case`, `CaseState`, `Action`, `Precondition`, `Postcondition`, `Dependency`, `PolicyRule`, `ValidationResult`, `Conflict`)
- [x] Sub-Evaluators (`PreconditionEvaluator`, `DependencyEvaluator`, `PolicyEvaluator`, `PostconditionConflictEvaluator`)
- [x] Deterministic Orchestration Engine (`DefaultValidationEngine`)
- [x] Semantic State Compiler & Deterministic Mock Domain Agents (`mosaic.compiler`, `mosaic.agents`)
- [x] Deterministic Intent & Evidence Decomposition Engine (`mosaic.intake`)
- [x] End-to-End Pipeline Orchestrator (`MosaicOrchestrator`)
- [x] Provider-Agnostic LLM Abstraction Layer & Mock Provider (`mosaic.llm`)
- [x] 33 passing pytest unit & end-to-end pipeline tests (`tests/`)

### Intentionally NOT Yet Implemented
- [ ] LLM Intent Decomposer (Phase 5B)
- [ ] Structured-Output Validation (Phase 5C)
- [ ] LLM -> MOSAIC Pipeline Integration (Phase 5D)
- [ ] Frontend Dashboard / UI
