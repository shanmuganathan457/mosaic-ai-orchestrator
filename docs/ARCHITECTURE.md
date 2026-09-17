# MOSAIC Architecture & Technical Specifications

**Multi-Intent Orchestration & State-Aware Intelligent Coordination**  
*Version 0.3.0 — Phase 3 Architecture & Compiler Specifications*

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

The system follows strict separation of concerns between domain perception/proposals, semantic state compilation, and validation logic.

```
Incoming Customer Communication (Email / Text)
                    │
                    ▼
          ┌───────────────────┐
          │   1. Intake API   │
          └─────────┬─────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 2. Intent & Evidence Compiler │
    └───────────────┬───────────────┘
                    │ (IntentSpans + Case Object)
                    ▼
    ┌─────────────────────────────────┐
    │ 3. Specialized Mock AI Agents   │
    │ (Security, Billing, Sub, Access)│  <-- "What do I propose?"
    └───────────────┬─────────────────┘
                    │ (Unvalidated AgentProposals)
                    ▼
   ┌─────────────────────────────────┐
   │ 4. Semantic State Compiler      │
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
     │ 5. Validation Outcome       │
     │    (ALLOW / BLOCK / ESCALATE)│
     └──────────────┬──────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    [ ALLOW ]          [ BLOCK / ESCALATE ]
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────────┐
│ 6. Response Synth│  │ 7. Human Escalation  │
│    & Execution   │  │    & Audit Logging   │
└──────────────────┘  └──────────────────────┘
```

### Component Responsibilities & Separation Boundaries

1. **Intake API & Case Management:** Normalizes incoming raw communications, establishes case tracking records, and tracks state transitions.
2. **Intent Decomposition Engine:** Parses compound customer messages into granular, evidence-backed `IntentSpan` records tied to verbatim text.
3. **Specialized Mock Domain Agents (`mosaic.agents`):** Domain-specific worker agents (Security, Billing, Subscription, Access) that produce unvalidated candidate `AgentProposal` payloads. *(Note: Currently implemented as deterministic mock agents for offline research benchmarking.)*
4. **Semantic State Compiler (`mosaic.compiler`):** Formats natural language / semi-structured proposals into strongly-typed `Action` primitives with explicit preconditions, postconditions, dependencies, and risk levels. **Crucially, the compiler performs zero validation decision making.**
5. **Deterministic Validation Engine (`mosaic.validation`):** The core research artifact. Evaluates structured actions against the current `CaseState`, active `Dependency` graphs, and system `PolicyRule`s strictly deterministically. Returns a `ValidationResult` (`ALLOW`, `BLOCK`, `ESCALATE`) along with explicit diagnostic `Conflict` items.
6. **Response Synthesizer / Execution Orchestrator:** Formulates final customer-facing responses using *only* explicitly `ALLOW`ed actions.
7. **Audit & Trace System:** Emits immutable `AuditEvent` logs capturing raw proposals, compiled actions, state snapshots, evaluation traces, and validation decisions.

---

## 3. Data Flow & Controlled Test Dataset

```
[Raw Case Text] ──► Mock Agents ──► AgentProposals ──► Compiler ──► Actions ──► Validator ──► Verdict
```

### Controlled Dataset (10 Benchmark Cases)
Located in `tests/fixtures/controlled_dataset.py`, covering:
1. **Case 1:** Single-intent valid subscription cancellation → `ALLOW`
2. **Case 2:** Single-intent refund request with unverified payment → `BLOCK`
3. **Case 3:** Two independent valid intents (Cancel + Refund) → `ALLOW`
4. **Case 4:** Access restoration with completed identity verification → `ALLOW`
5. **Case 5:** Access restoration without completed identity verification → `BLOCK`
6. **Case 6:** Security lock vs. Login access restoration conflict → `BLOCK`
7. **Case 7:** Access restoration with missing state fact → `BLOCK`
8. **Case 8:** Refund attempt under active `ACCOUNT_RESTRICTED` flag → `BLOCK`
9. **Case 9:** State-dependent refund (Unverified identity state) → `BLOCK`
10. **Case 10:** State-dependent refund (Verified identity state) → `ALLOW`

---

## 4. Technology Stack Decisions

- **Language:** Python 3.12+ (type safety, modern Pydantic v2 support, performance).
- **Web Framework:** FastAPI (async endpoints, OpenAPI spec integration, fast execution).
- **Data Validation & Schemas:** Pydantic v2 (strict validation, fast serialization).
- **Testing:** `pytest` for deterministic, offline testing of domain models, compiler, and validation engine.

---

## 5. Known Limitations & MVP Scope Boundaries

- **Mock Agents:** Domain agents are currently deterministic mock classes used to simulate multi-agent proposals without LLM latency or cost.
- **No Live CRM / Third-Party Integrations:** Production Salesforce/Zendesk APIs are mocked via localized Python interfaces.
- **Deterministic Pipeline:** LLM calls are strictly isolated from the core compiler and validation engine to maintain 100% testability and reproducibility.
