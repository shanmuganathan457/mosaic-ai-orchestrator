# MOSAIC Architecture & Technical Specifications

**Multi-Intent Orchestration & State-Aware Intelligent Coordination**  
*Version 0.5.0 — Phase 5A Architecture & LLM Abstraction Specifications*

---

## 1. Executive Summary & Research Positioning

MOSAIC is a research-oriented software architecture designed to investigate **Cross-Agent Semantic State Validation** within complex customer-service automation workflows.

### 1.1 Core Focus
Modern AI customer service pipelines commonly use multi-intent decomposition, RAG grounding, and multi-agent routing. However, when multiple specialized AI agents operate independently on distinct aspects of a compound customer inquiry, their individual proposed actions may be locally logical yet globally conflicting or policy-violating. 

MOSAIC introduces a **deterministic, inspectable, state-dependent Validation Engine** that compiles agent proposals into structured actions, checks preconditions, postconditions, workflow dependencies, and business policies, and outputs a validated verdict (`ALLOW`, `BLOCK`, or `ESCALATE`) **prior to response synthesis or side-effect execution**.

---

## 2. System Architecture & LLM Abstraction Boundary

The system maintains strict decoupling between non-deterministic LLM perception/generation models and deterministic validation logic.

```
                  Customer Email / Raw Message
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Provider      │
                    │ (Mock / Ollama /    │
                    │  Gemini / OpenAI)   │
                    └──────────┬──────────┘
                               │ (LLMRequest / LLMResponse)
                               ▼
                    ┌─────────────────────┐
                    │ LLM Abstraction     │
                    │ (BaseLLMProvider)   │
                    └──────────┬──────────┘
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

### Component Responsibilities & LLM Isolation

1. **LLM Abstraction Interface (`mosaic.llm`):** Abstract provider interface (`BaseLLMProvider`), request payload (`LLMRequest`), response container (`LLMResponse`), and exception definitions. Decouples MOSAIC from specific LLM provider SDKs (OpenAI, Gemini, Ollama, LiteLLM).
2. **Mock LLM Provider (`MockLLMProvider`):** 100% offline, deterministic implementation of `BaseLLMProvider` for research experiments and unit testing without network or GPU dependencies.
3. **Intent Decomposition Engine (`mosaic.intake`):** Rule-based or LLM-backed intent extractor parsing raw customer messages into non-overlapping `IntentSpan` objects and creating assigned `SubTask` records.
4. **Specialized Domain Agents (`mosaic.agents`):** Domain-specific worker agents (Security, Billing, Subscription, Access) producing unvalidated candidate `AgentProposal` payloads.
5. **Semantic State Compiler (`mosaic.compiler`):** Formats natural language / semi-structured proposals into strongly-typed `Action` primitives. Performs zero validation decision making.
6. **Deterministic Validation Engine (`mosaic.validation`):** Evaluates structured actions against current `CaseState`, `Dependency` graphs, and `PolicyRule`s strictly deterministically **without invoking an LLM**.

---

## 3. Technology Stack Decisions

- **Language:** Python 3.12+ (type safety, modern Pydantic v2 support, performance).
- **Web Framework:** FastAPI (async endpoints, OpenAPI spec integration, fast execution).
- **Data Validation & Schemas:** Pydantic v2 (strict validation, fast serialization).
- **LLM Abstraction Layer:** Custom provider-agnostic Pydantic/ABC boundary (`mosaic.llm`).
- **Testing:** `pytest` for deterministic, offline testing of domain models, compiler, intake engine, LLM abstraction, and validation engine.
