import React from 'react';
import type { ActionExecutionRecord } from '../types/mosaic';

interface ExecutionPanelProps {
  records: ActionExecutionRecord[];
}

export const ExecutionPanel: React.FC<ExecutionPanelProps> = ({ records }) => {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'SUCCESS':
        return 'badge-green';
      case 'BLOCKED':
        return 'badge-red';
      case 'ESCALATED':
        return 'badge-amber';
      case 'SKIPPED':
      default:
        return 'badge-purple';
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>⚙️</span> Backend Execution Log ({records.length})
      </div>
      {records.length === 0 ? (
        <div className="empty-state">No execution records</div>
      ) : (
        records.map((rec) => (
          <div key={rec.action_id} className="item-card">
            <div className="item-header">
              <span className="item-title">
                {rec.action_type} → {rec.target_entity_id}
              </span>
              <span className={`badge ${getStatusBadge(rec.status)}`}>
                {rec.status}
              </span>
            </div>
            <div className="item-details" style={{ marginBottom: '0.4rem' }}>
              Agent: <strong>{rec.agent_name}</strong> | Executed At: {new Date(rec.executed_at).toLocaleTimeString()}
            </div>
            {rec.message && (
              <div style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: '0.4rem' }}>
                {rec.message}
              </div>
            )}
            <pre className="code-block" style={{ color: '#bae6fd' }}>
              {JSON.stringify(rec.result_details, null, 2)}
            </pre>
          </div>
        ))
      )}
    </div>
  );
};
