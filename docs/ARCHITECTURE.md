# MOSAIC Architecture & Technical Specifications

**Multi-Intent Orchestration & State-Aware Intelligent Coordination**  
*Version 0.7.0 — Phase 5C Structured Semantic Action Compilation Specifications*

---

## 1. Executive Summary & Research Positioning

MOSAIC is a research-oriented software architecture designed to investigate **Cross-Agent Semantic State Validation** within complex customer-service automation workflows.

### 1.1 Core Focus
Modern AI customer service pipelines commonly use multi-intent decomposition, RAG grounding, and multi-agent routing. However, when multiple specialized AI agents operate independently on distinct aspects of a compound customer inquiry, their individual proposed actions may be locally logical yet globally conflicting or policy-violating. 

MOSAIC introduces a **deterministic, inspectable, state-dependent Validation Engine** that compiles agent proposals into structured actions, checks preconditions, postconditions, workflow dependencies, and business constraints, and outputs a validated verdict (`ALLOW`, `BLOCK`, or `ESCALATE`) **prior to response synthesis or side-effect execution**.

---

## 2. System Architecture & Output Integrity Enforcement

The architecture supports interchangeable intake strategies (`BaseIntentDecomposer`) and semantic compilation strategies (`BaseActionCompiler`), allowing direct experimental comparison between Deterministic semantic compilation and LLM-backed semantic compilation with strict boundary output integrity.

```
                      Customer Email / Raw Message
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
    DeterministicIntentDecomposer           LLMIntentDecomposer
    (Pattern-matching baseline)             (LLMProvider abstraction)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │ [Integrity Checked: Enum + Evidence + Metadata]
                                    ▼
                               IntentSpan[]
                                    │
                                    ▼
                                SubTask[]
                                    │
                                    ▼
                             Domain Agents (AgentProposal[])
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
     SemanticStateCompiler                   LLMActionCompiler
     (Deterministic baseline)               (LLMProvider abstraction)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │ [Traceability Preserved: proposal_id]
                                    ▼
                                 Action[]
                                    │
                                    ▼
                 ═════════════════════════════════════
                     MOSAIC VALIDATION ENGINE
                     (100% DETERMINISTIC, NO LLM)
                 ═════════════════════════════════════
                                    │
                            ┌───────┼───────┐
                            ▼       ▼       ▼
                          ALLOW   BLOCK  ESCALATE
```

### Output Integrity Principles (Phase 5B Quality Correction)
1. **Invalid Intent Category Handling:** If the LLM returns an unsupported or invalid `IntentCategory`, `LLMIntentDecomposer` raises an explicit `LLMResponseError`. Invalid model predictions are **never** silently repaired or defaulted to valid intents (e.g., `GENERAL_INQUIRY`).
2. **Evidence Traceability:** All extracted `verbatim_text` evidence spans must be strictly traceable to verbatim text in the customer message. Untraceable or hallucinated evidence spans raise an explicit `LLMResponseError`.
3. **Metadata Non-Fabrication:** The LLM prompt and decomposer enforce that entity identifiers (`payment_id`, `account_id`, `subscription_id`) are not fabricated if absent from customer text.

### Semantic Action Compilation & Decoupling (Phase 5C)
1. **Strategy Abstraction (`BaseActionCompiler`):** Standardized abstract contract converting an unvalidated `AgentProposal` into a strongly typed domain `Action`.
2. **Deterministic Baseline (`SemanticStateCompiler`):** Preserved baseline state compiler executing rule-based payload extraction without LLM calls.
3. **Structured Semantic Action Compilation (`LLMActionCompiler`):** Parses agent natural-language reasoning and raw payloads into `ExtractedActionPayload` Pydantic models using `BaseLLMProvider`.
4. **Strict Boundary Responsibility:** The LLM extracts action types, target entity IDs, preconditions, postconditions, dependencies, and risk levels. **The LLM NEVER evaluates case state, applies business policy, or determines ALLOW / BLOCK / ESCALATE verdicts.**
5. **Traceability:** Every compiled `Action` retains strict `proposal_id` linkage back to the generating `AgentProposal`.

---

## 3. Technology Stack Decisions

- **Language:** Python 3.12+ (type safety, modern Pydantic v2 support, performance).
- **Web Framework:** FastAPI (async endpoints, OpenAPI spec integration, fast execution).
- **Data Validation & Schemas:** Pydantic v2 (strict validation, fast serialization).
- **LLM Abstraction Layer:** Custom provider-agnostic Pydantic/ABC boundary (`mosaic.llm`).
- **Testing:** `pytest` for deterministic, offline testing of domain models, compiler, intake strategies, LLM abstraction, and validation engine.
