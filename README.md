# MOSAIC: Multi-Intent Orchestration & State-Aware Intelligent Coordination

> **A Research Architecture for Cross-Agent Semantic State Validation in Enterprise Customer Support Workflows**

---

## 1. What is MOSAIC?

MOSAIC is a research-oriented software architecture designed to evaluate **Cross-Agent Semantic State Validation**. 

When complex customer communications containing multiple intents (e.g., account compromised + subscription cancellation + refund request) are processed by independent specialized AI agents, individual proposed actions may be locally reasonable but globally conflicting, unsafe, or policy-violating.

MOSAIC introduces a **deterministic, inspectable, state-dependent Validation Engine** that compiles agent proposals into structured actions, checks preconditions, postconditions, dynamic case state facts, workflow dependencies, and business constraints, and outputs a validated verdict (`ALLOW`, `BLOCK`, or `ESCALATE`) **prior to response synthesis or side-effect execution**.

---

## 2. What MOSAIC Investigates vs. What It Is NOT

### Research Problem
Can a state/dependency-aware validation layer detect operational conflicts between independently generated AI actions before those actions reach the customer or an execution system?

### MOSAIC is NOT:
- Simply an email classifier
- A basic RAG chatbot
- A standard ticket router
- A generic multi-agent demo
- A parent/child ticket generator
- A keyword-matching contradiction detector

*Note:* Multi-intent extraction, ticket routing, RAG, and agent orchestration are established baseline capabilities. MOSAIC investigates the narrower validation control problem.

---

## 3. Current Implementation Status

- [x] **Phase 1: Project Initialization & Domain Schemas** (`docs/ARCHITECTURE.md`, `src/mosaic/domain/models/schemas.py`)
- [x] **Phase 2: Deterministic Validation Engine** (`src/mosaic/validation/`)
  - Sub-evaluators: `PreconditionEvaluator`, `DependencyEvaluator`, `PolicyEvaluator`, `PostconditionConflictEvaluator`
- [x] **Phase 3: Semantic State Compiler & Mock Domain Agents** (`src/mosaic/compiler/`, `src/mosaic/agents/`)
- [x] **Phase 4: Deterministic Intent Decomposition Engine & End-to-End Orchestrator** (`src/mosaic/intake/`, `src/mosaic/orchestrator.py`)
- [x] **Phase 5A: Provider-Agnostic LLM Abstraction Layer & Deterministic Mock LLM** (`src/mosaic/llm/`)
- [x] **Phase 5B: LLM-Backed Intent Decomposition Engine & Output Integrity Correction** (`src/mosaic/intake/llm_decomposer.py`)
- [x] **Phase 5C: Structured Semantic Action Compilation** (`src/mosaic/compiler/llm_compiler.py`)
- [x] **Phase 6: Research Dataset & Evaluation Harness + Integrity Correction** (`src/mosaic/evaluation/`, `tests/fixtures/research_dataset/v1/`)
  - Ground truth benchmark schema: `BenchmarkCase`, `EvaluationSystemResult`, `AggregateMetrics`
  - Synthetic dataset v1: 23 controlled scenarios (10 ALLOW, 10 BLOCK, 3 ESCALATED) covering missing preconditions, failed dependencies, cross-action postcondition conflicts, state-dependent execution, and security/compliance escalation
  - Baselines: Baseline A (Single-Intent Route), Baseline B (Direct Multi-Agent Synthesis without Validation), MOSAIC (State Validation)
  - Reusable evaluation harness: `EvaluationRunner` producing machine-readable `evaluation_results.json` reports
  - Objective research metrics: Verdict Accuracy, Micro-Aggregated Intent Precision/Recall/F1, Conflict Precision/Recall/F1, Escalation Precision, False Escalation Rate
- [x] **Phase 7A: Local LLM Provider Integration (Ollama)** (`src/mosaic/llm/ollama.py`, `src/mosaic/llm/factory.py`)
  - Local HTTP-based Ollama provider (`OllamaProvider`) conforming to `BaseLLMProvider` using zero external framework dependencies (`urllib.request`).
  - Provider Factory (`LLMProviderFactory`) for runtime resolution of `mock` and `ollama` providers.
- [x] **Phase 7B: Natural-Language LLM Evaluation** (`tests/fixtures/research_dataset/v1_natural_language/`, `src/mosaic/evaluation/`)
  - Synthetic Natural Language Benchmark Dataset (`v1_natural_language`): 39 natural-language variants referencing authoritative `source_case_id` ground truth.
  - Action-level metrics: Action Coverage, Unexpected Action Rate.
  - Root Cause Error Attribution Taxonomy: `CORRECT_ALL`, `INCORRECT_INTENT_EXTRACTION`, `INCORRECT_ACTION_COMPILATION`, `INCORRECT_VALIDATION_RESULT`, `INCORRECT_FINAL_INTERPRETATION`.
  - Reusable CLI Entry Point (`python -c "import sys; sys.path.insert(0, 'src'); from mosaic.evaluation.runner import main; main()" --provider mock --dataset v1_natural_language`).
  - Serialized evaluation reports output to `evaluation_results/phase_7b/<model>/`.
  - 72 passing unit/integration tests + 1 optional live Ollama skipped test (`tests/`).

---

## 4. System Architecture & Data Flow

```
Incoming Customer Communication (Raw Message)
              │
    ┌─────────┴─────────┐
    ▼                   ▼
Deterministic         LLM Intent
Decomposer            Decomposer
    │                   │
    └─────────┬─────────┘
              │ (BaseIntentDecomposer strategy)
              ▼
   1. Intent & Evidence Spans (IntentSpans)
              │
              ▼
  2. Specialized Mock Domain Agents (AgentProposals)
     (Security, Billing, Subscription, Access)
              │
              ▼
  3. Semantic State Compiler (Structured Actions)
              │
              ▼
════════════════════════════════════════════════════
 DETERMINISTIC VALIDATION ENGINE (NO LLM REQUIRED)
 - Precondition Evaluation
 - Dependency Verification
 - Policy Constraint Evaluation
 - Postcondition Conflict Detection
════════════════════════════════════════════════════
              │
              ├─────────────────────────┐
              ▼                         ▼
        [ ALLOW ]              [ BLOCK / ESCALATE ]
              │                         │
              ▼                         ▼
   4. Response Synthesis &     5. Human Escalation &
      Side-Effect Execution       Audit Logging
```

---

## 5. Repository Structure

```
mosaic-ai-orchestrator/
├── docs/
│   ├── ARCHITECTURE.md          # Architectural specifications & Phase 5B LLM intake
│   └── RESEARCH_ALIGNMENT.md    # Mapping code modules to research document
├── src/
│   └── mosaic/
│       ├── agents/              # Deterministic Mock Domain Agents
│       │   ├── __init__.py
│       │   └── mock_agents.py
│       ├── compiler/            # Semantic State Compiler (AgentProposal -> Action)
│       │   ├── __init__.py
│       │   └── state_compiler.py
│       ├── config/              # Pydantic-based configuration management
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── domain/
│       │   ├── models/          # Pydantic v2 core schemas & domain models
│       │   │   ├── __init__.py
│       │   │   └── schemas.py
│       │   └── __init__.py
│       ├── intake/              # Intake Engines (Deterministic & LLM Strategy)
│       │   ├── __init__.py
│       │   ├── decomposer.py
│       │   └── llm_decomposer.py
│       ├── llm/                 # Provider-Agnostic LLM Abstraction Layer
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── mock.py
│       │   └── models.py
│       ├── validation/          # Core deterministic Validation Engine & sub-evaluators
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── engine.py
│       │   └── evaluators.py
│       ├── __init__.py
│       ├── main.py              # FastAPI entrypoint & health check endpoint
│       └── orchestrator.py      # End-to-end pipeline coordinator
├── tests/
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── controlled_dataset.py# 10-Case Controlled Test Dataset
│   ├── __init__.py
│   ├── test_compiler_and_agents.py
│   ├── test_domain_models.py
│   ├── test_end_to_end_pipeline.py
│   ├── test_intake_and_orchestrator.py
│   ├── test_llm_abstraction.py
│   ├── test_llm_decomposer.py
│   └── test_validation_engine.py
├── pyproject.toml               # Build & dependency configuration
└── README.md                    # Project documentation
```

---

## 6. Setup & How to Run

### Prerequisites
- Python 3.12+ installed

### Environment Setup

1. **Clone repository and enter project directory:**
   ```bash
   cd mosaic-ai-orchestrator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run Complete Pytest Suite (43 Tests):**
   ```bash
   python -m pytest
   ```

5. **Start FastAPI Development Server:**
   ```bash
   uvicorn mosaic.main:app --reload
   ```
   Access health check endpoint at `http://127.0.0.1:8000/health`.
