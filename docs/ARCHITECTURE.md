# MOSAIC Architecture & Technical Specifications

**Multi-Intent Orchestration & State-Aware Intelligent Coordination**  
*Version 0.4.0 — Phase 4 Architecture & Intent Decomposition Specifications*

---

## 1. Executive Summary & Research Positioning

MOSAIC is a research-oriented software architecture designed to investigate **Cross-Agent Semantic State Validation** within complex customer-service automation workflows.

### 1.1 Core Focus
Modern AI customer service pipelines commonly use multi-intent decomposition, RAG grounding, and multi-agent routing. However, when multiple specialized AI agents operate independently on distinct aspects of a compound customer inquiry, their individual proposed actions may be locally logical yet globally conflicting or policy-violating. 

MOSAIC introduces a **deterministic, inspectable, state-dependent Validation Engine** that compiles agent proposals into structured actions, checks preconditions, postconditions, workflow dependencies, and business policies, and outputs a validated verdict (`ALLOW`, `BLOCK`, or `ESCALATE`) **prior to response synthesis or side-effect execution**.

### 1.2 Baseline Capabilities vs. Research Scope
- **Baseline Capabilities (Non-novel / Existing):** Multi-intent extraction, ticket routing, RAG grounding, basic multi-agent orchestration, static keyword/pairwise contradiction detection.
- **MOSAIC Research Contribution:** State/dependency-aware semantic validation layer evaluating cross-agent business actions against dynamic case state and workflow policy constraints without relying on non-deterministic LLM calls during validation.

---

## 2. System Architecture & Component Boundaries

The system follows strict separation of concerns between raw intake perception, domain agent proposals, semantic state compilation, and validation logic.

```
Incoming Customer Communication (Raw Case Message)
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 1. Intent Decomposition Engine│  <-- "Extract IntentSpans & Create SubTasks"
    │    (mosaic.intake)            │
    └───────────────┬───────────────┘
                    │ (Canonical Case + SubTasks)
                    ▼
    ┌─────────────────────────────────┐
    │ 2. Specialized Mock AI Agents   │
    │ (Security, Billing, Sub, Access)│  <-- "What do I propose?"
    └───────────────┬─────────────────┘
                    │ (Unvalidated AgentProposals)
                    ▼
   ┌─────────────────────────────────┐
   │ 3. Semantic State Compiler      │
   │    (Translates proposals into   │  <-- "What structured action does this represent?"
   │     strongly typed Actions)     │
   └────────────────┬────────────────┘
                    │ (Structured Action Primitives)
                    ▼
══════════════════════════════════════════════════════════════
    DETERMINISTIC VALIDATION ENGINE (NO LLM REQUIRED)
    <-- "Is that action safe/valid in the current state?"
══════════════════════════════════════════════════════════════
                    │
                    ├─► Precondition Evaluation
                    ├─► State & Dependency Verification
                    ├─► Policy Rule & Constraint Check
                    └─► Cross-Action Conflict Detection
                    │
                    ▼
     ┌─────────────────────────────┐
     │ 4. Validation Outcome       │
     │    (ALLOW / BLOCK / ESCALATE)│
     └──────────────┬──────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    [ ALLOW ]          [ BLOCK / ESCALATE ]
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────────┐
│ 5. Response Synth│  │ 6. Human Escalation  │
│    & Execution   │  │    & Audit Logging   │
└──────────────────┘  └──────────────────────┘
```

### Component Responsibilities & Separation Boundaries

1. **Intent Decomposition Engine (`mosaic.intake`):** Rule-based, deterministic intent extractor parsing raw customer messages into non-overlapping `IntentSpan` objects and creating assigned `SubTask` records.
2. **Specialized Mock Domain Agents (`mosaic.agents`):** Domain-specific worker agents (Security, Billing, Subscription, Access) that produce unvalidated candidate `AgentProposal` payloads based on assigned subtasks.
3. **Semantic State Compiler (`mosaic.compiler`):** Formats natural language / semi-structured proposals into strongly-typed `Action` primitives with explicit preconditions, postconditions, dependencies, and risk levels. **Crucially, the compiler performs zero validation decision making.**
4. **Deterministic Validation Engine (`mosaic.validation`):** Evaluates structured actions against the current `CaseState`, active `Dependency` graphs, and system `PolicyRule`s strictly deterministically. Returns a `ValidationResult` (`ALLOW`, `BLOCK`, `ESCALATE`) along with explicit diagnostic `Conflict` items.
5. **Mosaic Orchestrator (`mosaic.orchestrator`):** Pipeline manager coordinating end-to-end flow from Raw Intake to Final Validation Verdict.

---

## 3. Technology Stack Decisions

- **Language:** Python 3.12+ (type safety, modern Pydantic v2 support, performance).
- **Web Framework:** FastAPI (async endpoints, OpenAPI spec integration, fast execution).
- **Data Validation & Schemas:** Pydantic v2 (strict validation, fast serialization).
- **Testing:** `pytest` for deterministic, offline testing of domain models, compiler, intake engine, and validation engine.
