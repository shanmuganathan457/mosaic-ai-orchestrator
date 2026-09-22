import React from 'react';
import type { GovernanceExecutionResponse } from '../types/mosaic';

interface HistoryPanelProps {
  history: GovernanceExecutionResponse[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({ history, selectedIndex, onSelect }) => {
  if (history.length === 0) return null;

  return (
    <div className="card">
      <div className="card-title">
        <span>📜</span> Execution History ({history.length})
      </div>
      <div className="history-list">
        {history.map((item, idx) => (
          <div
            key={item.request_id}
            className={`history-item ${selectedIndex === idx ? 'active' : ''}`}
            onClick={() => onSelect(idx)}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="history-text">{item.raw_message}</div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                {item.customer_id} • {new Date(item.processed_at).toLocaleTimeString()}
              </div>
            </div>
            <span
              className={`preset-tag ${
                item.final_verdict === 'ALLOW'
                  ? 'allow'
                  : item.final_verdict === 'BLOCK'
                  ? 'block'
                  : 'escalate'
              }`}
              style={{ marginLeft: '0.5rem' }}
            >
              {item.final_verdict}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
