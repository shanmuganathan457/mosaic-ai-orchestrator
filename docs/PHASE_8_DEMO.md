# MOSAIC: Governance & Multi-Intent AI Orchestrator

## 1. What MOSAIC Does

MOSAIC (**Multi-Intent Orchestration & State-Aware Intelligent Coordination**) is a deterministic governance and orchestration layer designed to sit between incoming AI/customer support requests and sensitive backend enterprise APIs.

Rather than allowing an LLM to execute actions directly or make unvalidated side-effects, MOSAIC enforces a multi-stage safety pipeline:

1. **Intent Decomposition**: Extracts discrete customer intents and grounds them to verbatim text evidence.
2. **Semantic Action Compilation**: Maps intents into structured semantic actions with assigned risk levels and target entities.
3. **Deterministic Validation**: Evaluates compiled actions against strict domain policy rules, state facts, prerequisites, and cross-action dependency graphs.
4. **Safety Verdict Enforcement**: Outputs a binding verdict (`ALLOW`, `BLOCK`, or `ESCALATED`).
5. **Backend Execution Control**: Executes actions *only* if the verdict is `ALLOW`. Any request resulting in `BLOCK` or `ESCALATED` is prevented from triggering side-effects.

---

## 2. Architecture Flow

```text
Customer Support Request / User UI
              │
              ▼
   FastAPI Intake API Endpoint (`POST /api/v1/support/requests`)
              │
              ▼
  ┌────────────────────────────────────────────────────────┐
  │                   MOSAIC ENGINE                        │
  │                                                        │
  │  1. Intent Decomposer (Rule/LLM Strategy)               │
  │     └── Extracts Intent Categories & Verbatim Text     │
  │                                                        │
  │  2. Action Compiler                                    │
  │     └── Maps Intents -> Structured Semantic Actions    │
  │                                                        │
  │  3. Deterministic Validation Engine                     │
  │     ├── Checks Preconditions & State Facts             │
  │     ├── Enforces Policy Constraints & Flags            │
  │     └── Resolves Cross-Action Conflicts/Dependencies   │
  └────────────────────────────────────────────────────────┘
              │
              ▼
     Safety Verdict Output
  ┌───────────┬───────────┬─────────────┐
  │   ALLOW   │   BLOCK   │  ESCALATED  │
  └─────┬─────┴─────┬─────┴──────┬──────┘
        │           │            │
        ▼           ▼            ▼
  Mock Executor   No Execution  Human Queue
   (Executed)   (Refused)     (Escalated)
        │           │            │
        └───────────┼────────────┘
                    ▼
           React UI Dashboard
```

---

## 3. Startup Commands

### Backend Startup (FastAPI + Uvicorn)

From the project root:

```bash
# Ensure Python dependencies are installed
pip install -e .

# Run the FastAPI server
python src/mosaic/main.py
# Server starts on http://localhost:8000
```

### Frontend Startup (Vite + React + TypeScript)

From the `frontend/` directory:

```bash
cd frontend

# Install node dependencies
npm install

# Start Vite development server
npm run dev
# Dashboard available at http://localhost:5173
```

---

## 4. API Endpoint

- **URL**: `POST http://localhost:8000/api/v1/support/requests`
- **Headers**: `Content-Type: application/json`
- **Payload Schema**:
  ```json
  {
    "customer_id": "cust_101",
    "raw_message": "Please refund payment pay_101.",
    "initial_facts": {
      "payment_verified": true,
      "identity_verified": true
    },
    "active_flags": []
  }
  ```

---

## 5. Environment Configuration

The frontend references `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 6. Demonstration Scenarios & Observed Behavior

### Scenario A — ALLOW (Valid Refund Request)
- **Input**: Message `"Please refund payment pay_101."` with `initial_facts={"payment_verified": true, "identity_verified": true}`.
- **Observed Behavior**:
  - Intent Detected: `BILLING` / `refund_payment` (Confidence: 95%).
  - Action Compiled: `refund_payment` on target `pay_101` (Risk: `HIGH`).
  - Validation Verdict: `ALLOW` (0 conflicts detected).
  - Execution Status: `EXECUTED` via `MockExecutor` (`BillingMockAgent`).
  - UI Display: Green `ALLOW` badge + `EXECUTED` status.

### Scenario B — BLOCK (Unverified Refund Request)
- **Input**: Message `"Please process a refund for pay_102 right now!"` with `initial_facts={"payment_verified": true, "identity_verified": false}`.
- **Observed Behavior**:
  - Intent Detected: `BILLING` / `refund_payment`.
  - Action Compiled: `refund_payment` on target `pay_102`.
  - Primary Conflict Detected: `PRECONDITION_UNSATISFIED` (`PRECONDITION_FAILED`: Requires `identity_verified=True`, current state is `False`).
  - Validation Verdict: `BLOCK`.
  - Execution Status: `BLOCKED` (Mock executor refused execution).
  - UI Display: Red `BLOCK` badge + `BLOCKED` status.

### Scenario C — ESCALATE (Suspicious Location Login Recovery)
- **Input**: Message `"Please restore login access for my account acc_707."` with `initial_facts={"account_status": "locked"}` and `active_flags=["SUSPICIOUS_LOCATION_LOGIN"]`.
- **Observed Behavior**:
  - Intent Detected: `ACCESS_RESTORATION` / `restore_login_access`.
  - Action Compiled: `restore_login_access` on target `acc_707`.
  - Primary Conflict Detected: `AMBIGUOUS_EVIDENCE` (`POLICY_FORBIDDEN_FLAG`: Policy `pol_access_suspicious_location_escalate` triggered due to active flag `SUSPICIOUS_LOCATION_LOGIN`).
  - Validation Verdict: `ESCALATED`.
  - Execution Status: `ESCALATED` (Marked for human review queue, zero backend execution).
  - UI Display: Amber `ESCALATED` badge + `ESCALATED` status.

### Scenario D — MULTI-INTENT (Subscription Cancellation + Refund)
- **Input**: Message `"Cancel my active subscription sub_202 and also refund pay_101."` with `initial_facts={"subscription_active": true, "payment_verified": true, "identity_verified": true}`.
- **Observed Behavior**:
  - Intents Detected: `BILLING` / `refund_payment` AND `SUBSCRIPTION` / `cancel_subscription`.
  - Actions Compiled: `refund_payment` (`pay_101`) AND `cancel_subscription` (`sub_202`).
  - Validation Verdict: `ALLOW` (Both action preconditions satisfied).
  - Execution Status: `EXECUTED` (Both actions executed independently by respective agents).
  - UI Display: Green `ALLOW` badge showing both detected intents, actions, and execution records.

---

## 7. Explanation of Safety Boundaries

MOSAIC strictly enforces execution control based on verdict:

| Verdict | Governance Behavior | Backend Execution |
| :--- | :--- | :--- |
| **`ALLOW`** | All state preconditions, policies, and dependencies passed. | **Executed** by backend agents. |
| **`BLOCK`** | Fatal precondition or safety policy violation. | **NEVER Executed**. State remains unchanged. |
| **`ESCALATED`** | Ambiguous policy state or required human-in-the-loop verification. | **NEVER Executed**. Routed to human agent review. |

This boundary guarantees that unvalidated, untrusted LLM outputs can never trigger unintended side-effects in enterprise backends.

---

## 8. Integration Blueprint for Production Deployment

Currently, MOSAIC utilizes a **Mock Backend Executor** (`MockExecutor`) for demonstration purposes. 

To deploy MOSAIC in a production environment, an enterprise can connect real internal systems without modifying the core MOSAIC governance engine:

1. **Account / Authentication System**:
   - Replace mock identity facts with live lookup calls to Okta, Auth0, or internal IAM services.
2. **Payment / Refund API**:
   - Bind the `refund_payment` agent handler to Stripe, PayPal, or internal billing microservices.
3. **Subscription Management**:
   - Bind the `cancel_subscription` agent handler to Chargebee, Recurly, or internal subscription APIs.
4. **CRM & Support Ticket System**:
   - Connect `ESCALATED` verdicts to trigger automated Zendesk, Salesforce Service Cloud, or ServiceNow ticket creation for human agent takeover.

> **Note**: MOSAIC functions exclusively as the **governance and orchestration layer**. It does not replace existing CRM, IAM, or payment infrastructure; rather, it safely orchestrates requests before they touch those systems.
