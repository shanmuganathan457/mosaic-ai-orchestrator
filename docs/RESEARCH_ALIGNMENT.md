# MOSAIC Research Alignment Matrix

This document maps software components in the **MOSAIC** codebase to the research concepts, requirements, and hypotheses specified in `docs/MOSAIC_Research_Document.docx`.

---

## 1. Research Objectives & System Mapping

| Research Concept / Requirement | Software Component / Module | Implementation Purpose |
| :--- | :--- | :--- |
| **LLM-Backed Intent & Evidence Extraction** | `mosaic.intake.llm_decomposer.LLMIntentDecomposer` | LLM-backed strategy converting raw text into structured `IntentSpan` objects via `BaseLLMProvider`. |
| **Deterministic Intent Extraction Baseline** | `mosaic.intake.decomposer.IntentDecompositionEngine` | Rule-based baseline intent extractor for empirical comparability against LLM extraction. |
| **Intake Strategy Abstraction** | `mosaic.intake.decomposer.BaseIntentDecomposer` | Strategy interface ensuring interchangeable intake components feeding identical downstream `IntentSpan[]` representations. |
| **Provider-Agnostic LLM Abstraction** | `mosaic.llm.base.BaseLLMProvider`, `mosaic.llm.models` | Establishes a generic Pydantic/ABC boundary (`LLMRequest`, `LLMResponse`) decoupling MOSAIC from provider SDKs. |
| **Deterministic Research Mock LLM** | `mosaic.llm.mock.MockLLMProvider` | Provides a 100% offline, reproducible mock provider for research benchmarking without API/GPU costs. |
| **Case & Sub-Task Tracking** | `mosaic.domain.models.schemas` (`Case`, `SubTask`, `CaseState`) | Maintains state records, current facts, active flags, and completed action graphs for multi-intent customer communications. |
| **Agent Action Proposals** | `mosaic.agents.mock_agents` | Deterministic mock worker agents generating `AgentProposal` objects. |
| **Semantic Action Compilation Strategy** | `mosaic.compiler.state_compiler.BaseActionCompiler` | Abstract Strategy interface ensuring interchangeable semantic action compilers feeding identical downstream `Action[]` primitives. |
| **Deterministic Action Compiler Baseline** | `mosaic.compiler.state_compiler.SemanticStateCompiler` | Rule-based baseline compiler converting proposals into structured actions for empirical comparability. |
| **LLM-Backed Action Compiler** | `mosaic.compiler.llm_compiler.LLMActionCompiler` | Structured LLM-backed compiler transforming agent proposals into validated domain `Action` objects via `BaseLLMProvider`. |
| **State & Policy Validation** | `mosaic.validation.engine.DefaultValidationEngine` | Evaluates structured actions against case facts, preconditions, postconditions, and business constraints deterministically without calling an LLM. |
| **End-to-End Pipeline Coordination** | `mosaic.orchestrator.MosaicOrchestrator` | Coordinates raw text intake, mock agent execution, state compilation, and validation evaluation. |

---

## 2. Research Alignment Details

### 2.1 Strategy Comparability (Deterministic vs. LLM)
Research requirement: Evaluate LLM perception accuracy against deterministic baselines without altering downstream validation.
- Both `IntentDecompositionEngine` and `LLMIntentDecomposer` implement `BaseIntentDecomposer`.
- Both `SemanticStateCompiler` and `LLMActionCompiler` implement `BaseActionCompiler`.
- Both yield identical `IntentSpan[]`, `SubTask[]`, and `Action[]` objects fed into domain agents and the `DefaultValidationEngine`.

### 2.2 LLM Perception vs. Validation Decoupling
Research requirement: The LLM may propose candidate intents or actions, but **must never make the final safety or validation decision**.
- The `LLMIntentDecomposer` is restricted to intent, verbatim evidence, and confidence extraction.
- The `LLMActionCompiler` is restricted to structured semantic action parameter and condition extraction.
- The `DefaultValidationEngine` in `mosaic.validation` remains 100% deterministic and LLM-independent.

---

## 3. Implementation Status Summary

### Currently Implemented
- [x] Pydantic Domain Schemas (`Case`, `CaseState`, `Action`, `Precondition`, `Postcondition`, `Dependency`, `PolicyRule`, `ValidationResult`, `Conflict`)
- [x] Sub-Evaluators (`PreconditionEvaluator`, `DependencyEvaluator`, `PolicyEvaluator`, `PostconditionConflictEvaluator`)
- [x] Deterministic Orchestration Engine (`DefaultValidationEngine`)
- [x] Semantic State Compiler & Deterministic Mock Domain Agents (`mosaic.compiler`, `mosaic.agents`)
- [x] Deterministic Intent & Evidence Decomposition Engine (`mosaic.intake.decomposer`)
- [x] Provider-Agnostic LLM Abstraction Layer & Mock Provider (`mosaic.llm`)
- [x] Local LLM Provider Integration via Ollama (`mosaic.llm.ollama.OllamaProvider`, `mosaic.llm.factory.LLMProviderFactory`)
- [x] Cloud LLM Provider Integration via Google Gemini SDK (`mosaic.llm.gemini.GeminiProvider`, `google-genai`)
- [x] LLM-Backed Intent & Evidence Decomposition Engine (`mosaic.intake.llm_decomposer`)
- [x] Abstract Action Compiler Strategy (`BaseActionCompiler`)
- [x] LLM-Backed Structured Semantic Action Compiler (`LLMActionCompiler`)
- [x] Controlled Research Benchmark Dataset v1 (`tests/fixtures/research_dataset/v1/cases.json`)
- [x] Natural Language Benchmark Dataset v1_natural_language (`tests/fixtures/research_dataset/v1_natural_language/cases.json`)
- [x] Root Cause Error Attribution Taxonomy Evaluator (`mosaic.evaluation.error_attribution`)
- [x] Research Baseline Implementations (`BaselineASingleIntentSystem`, `BaselineBDirectMultiAgentSystem`)
- [x] Evaluation Harness Runner, CLI & Metrics (`EvaluationRunner`, `compute_aggregate_metrics`)
- [x] Strategy-Injectable Pipeline Orchestrator (`MosaicOrchestrator`)
- [x] 81 passing pytest unit, integration, & benchmark runner tests (`tests/`)

### Intentionally NOT Yet Implemented
- [ ] Response Synthesis & Real System Execution (Phase 8)
- [ ] Frontend Dashboard / UI
