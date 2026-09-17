# MOSAIC Architecture & Technical Specifications

**Multi-Intent Orchestration & State-Aware Intelligent Coordination**  
*Version 0.6.0 — Phase 5B Architecture & LLM Intent Decomposition Specifications*

---

## 1. Executive Summary & Research Positioning

MOSAIC is a research-oriented software architecture designed to investigate **Cross-Agent Semantic State Validation** within complex customer-service automation workflows.

### 1.1 Core Focus
Modern AI customer service pipelines commonly use multi-intent decomposition, RAG grounding, and multi-agent routing. However, when multiple specialized AI agents operate independently on distinct aspects of a compound customer inquiry, their individual proposed actions may be locally logical yet globally conflicting or policy-violating. 

MOSAIC introduces a **deterministic, inspectable, state-dependent Validation Engine** that compiles agent proposals into structured actions, checks preconditions, postconditions, workflow dependencies, and business policies, and outputs a validated verdict (`ALLOW`, `BLOCK`, or `ESCALATE`) **prior to response synthesis or side-effect execution**.

---

## 2. System Architecture & Intake Strategy Pattern

The architecture supports interchangeable intake strategies (`BaseIntentDecomposer`), allowing direct experimental comparison between Rule-Based Deterministic Extraction and LLM-Backed Intent Extraction.

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
                                    │
                                    ▼
                               IntentSpan[]
                                    │
                                    ▼
                                SubTask[]
                                    │
                                    ▼
                             Domain Agents
                                    │
                                    ▼
                           Semantic Compiler
                                    │
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

### Component Responsibilities & LLM Boundaries

1. **Intake Strategy Boundary (`BaseIntentDecomposer`):** Strategy interface for intent extraction. Guarantees that both `IntentDecompositionEngine` (rule-based) and `LLMIntentDecomposer` (LLM-backed) output identical `IntentSpan[]` representations.
2. **LLM Intent Decomposer (`LLMIntentDecomposer`):** Uses `BaseLLMProvider` to extract intents, verbatim evidence spans, and confidence scores. Uses strict Pydantic boundary schemas (`ExtractedIntentsPayload`) to validate output formatting. **The LLM is strictly prohibited from making validation decisions (`ALLOW`, `BLOCK`, `ESCALATE`).**
3. **Specialized Domain Agents (`mosaic.agents`):** Domain-specific worker agents (Security, Billing, Subscription, Access) producing unvalidated candidate `AgentProposal` payloads based on assigned subtasks.
4. **Semantic State Compiler (`mosaic.compiler`):** Formats natural language / semi-structured proposals into strongly-typed `Action` primitives. Performs zero validation decision making.
5. **Deterministic Validation Engine (`mosaic.validation`):** Evaluates structured actions against current `CaseState`, `Dependency` graphs, and `PolicyRule`s strictly deterministically **without invoking an LLM**.

---

## 3. Technology Stack Decisions

- **Language:** Python 3.12+ (type safety, modern Pydantic v2 support, performance).
- **Web Framework:** FastAPI (async endpoints, OpenAPI spec integration, fast execution).
- **Data Validation & Schemas:** Pydantic v2 (strict validation, fast serialization).
- **LLM Abstraction Layer:** Custom provider-agnostic Pydantic/ABC boundary (`mosaic.llm`).
- **Testing:** `pytest` for deterministic, offline testing of domain models, compiler, intake strategies, LLM abstraction, and validation engine.
