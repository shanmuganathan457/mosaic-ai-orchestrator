import React from 'react';
import type { ProposedAction } from '../types/mosaic';

interface ActionListProps {
  actions: ProposedAction[];
}

export const ActionList: React.FC<ActionListProps> = ({ actions }) => {
  const getRiskBadge = (risk: string) => {
    switch (risk.toUpperCase()) {
      case 'HIGH':
        return 'badge-red';
      case 'MEDIUM':
        return 'badge-amber';
      default:
        return 'badge-blue';
    }
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>⚡</span> Proposed Semantic Actions ({actions.length})
      </div>
      {actions.length === 0 ? (
        <div className="empty-state">No actions proposed</div>
      ) : (
        actions.map((action) => (
          <div key={action.id} className="item-card">
            <div className="item-header">
              <span className="item-title">{action.action_type}</span>
              <span className={`badge ${getRiskBadge(action.risk_level)}`}>
                {action.risk_level} RISK
              </span>
            </div>
            <div className="item-details" style={{ marginBottom: '0.4rem' }}>
              Target: <strong>{action.target_entity_id}</strong> | Agent: <strong>{action.agent_name}</strong>
            </div>
            <pre className="code-block">
              {JSON.stringify(action.parameters, null, 2)}
            </pre>
          </div>
        ))
      )}
    </div>
  );
};
