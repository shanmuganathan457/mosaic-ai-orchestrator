import React from 'react';

interface VerdictBadgeProps {
  verdict: 'ALLOW' | 'BLOCK' | 'ESCALATED';
  explanation: string;
  executionStatus: string;
}

export const VerdictBadge: React.FC<VerdictBadgeProps> = ({ verdict, explanation, executionStatus }) => {
  const getIcon = () => {
    switch (verdict) {
      case 'ALLOW':
        return '✓';
      case 'BLOCK':
        return '✕';
      case 'ESCALATED':
        return '⚠';
    }
  };

  return (
    <div className={`verdict-banner ${verdict}`}>
      <div>
        <div className="verdict-title">
          <span>{getIcon()}</span>
          <span>{verdict}</span>
        </div>
        <div className="verdict-explanation">{explanation}</div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: '0.75rem', opacity: 0.8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Execution Status
        </div>
        <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>
          {executionStatus}
        </div>
      </div>
    </div>
  );
};
