import { useState } from 'react';
import { submitSupportRequest } from './api/mosaicApi';
import type { GovernanceExecutionResponse, SupportRequestInput } from './types/mosaic';
import { RequestForm } from './components/RequestForm';
import { VerdictBadge } from './components/VerdictBadge';
import { IntentList } from './components/IntentList';
import { ActionList } from './components/ActionList';
import { ConflictList } from './components/ConflictList';
import { ExecutionPanel } from './components/ExecutionPanel';
import { HistoryPanel } from './components/HistoryPanel';

export function App() {
  const [history, setHistory] = useState<GovernanceExecutionResponse[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFormSubmit = async (input: SupportRequestInput) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await submitSupportRequest(input);
      setHistory((prev) => [response, ...prev]);
      setSelectedIndex(0);
    } catch (err: any) {
      setError(err.message || 'Failed to communicate with MOSAIC API.');
    } finally {
      setIsLoading(false);
    }
  };

  const currentResult = selectedIndex !== null ? history[selectedIndex] : null;

  return (
    <div>
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">M</div>
          <div>
            <span className="brand-title">MOSAIC</span>
            <span className="brand-subtitle">AI Governance & Multi-Intent Orchestrator</span>
          </div>
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Backend Status: <span style={{ color: '#34d399', fontWeight: 600 }}>● Online (localhost:8000)</span>
        </div>
      </header>

      <main className="app-layout">
        <aside className="sidebar">
          <RequestForm onSubmit={handleFormSubmit} isLoading={isLoading} />
          <HistoryPanel
            history={history}
            selectedIndex={selectedIndex}
            onSelect={(idx) => setSelectedIndex(idx)}
          />
        </aside>

        <section className="main-dashboard">
          {error && (
            <div className="card" style={{ borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
              <div style={{ color: '#f87171', fontWeight: 600, marginBottom: '0.25rem' }}>⚠️ API Error</div>
              <div style={{ color: '#fca5a5', fontSize: '0.9rem' }}>{error}</div>
            </div>
          )}

          {!currentResult && !error && (
            <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🧩</div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                MOSAIC Pipeline Standby
              </h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '480px', margin: '0 auto 1.5rem auto' }}>
                Select a preset test scenario or submit a customer support request on the left to see intent detection, deterministic validation, and backend action execution in real time.
              </p>
            </div>
          )}

          {currentResult && (
            <>
              <VerdictBadge
                verdict={currentResult.final_verdict}
                explanation={currentResult.human_readable_explanation}
                executionStatus={currentResult.execution_status}
              />

              <div className="grid-two">
                <IntentList intents={currentResult.detected_intents} />
                <ActionList actions={currentResult.proposed_actions} />
              </div>

              <ConflictList
                primaryConflicts={currentResult.primary_conflicts}
                secondaryConflicts={currentResult.secondary_conflicts}
              />

              <ExecutionPanel records={currentResult.action_execution_records} />
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
