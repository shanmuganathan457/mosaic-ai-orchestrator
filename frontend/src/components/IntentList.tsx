import React from 'react';
import type { Intent } from '../types/mosaic';

interface IntentListProps {
  intents: Intent[];
}

export const IntentList: React.FC<IntentListProps> = ({ intents }) => {
  return (
    <div className="card">
      <div className="card-title">
        <span>🎯</span> Detected Intents ({intents.length})
      </div>
      {intents.length === 0 ? (
        <div className="empty-state">No intents detected</div>
      ) : (
        intents.map((intent) => (
          <div key={intent.id} className="item-card">
            <div className="item-header">
              <span className="item-title">{intent.intent_name}</span>
              <span className="badge badge-purple">
                {(intent.confidence * 100).toFixed(0)}% Confidence
              </span>
            </div>
            <div className="item-details" style={{ marginBottom: '0.3rem' }}>
              Category: {intent.category}
            </div>
            <div className="code-block" style={{ color: '#93c5fd' }}>
              "{intent.verbatim_text}"
            </div>
          </div>
        ))
      )}
    </div>
  );
};
