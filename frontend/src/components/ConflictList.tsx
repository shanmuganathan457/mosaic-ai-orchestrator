import React from 'react';
import type { ConflictRecord } from '../types/mosaic';

interface ConflictListProps {
  primaryConflicts: ConflictRecord[];
  secondaryConflicts: ConflictRecord[];
}

export const ConflictList: React.FC<ConflictListProps> = ({ primaryConflicts, secondaryConflicts }) => {
  const totalConflicts = primaryConflicts.length + secondaryConflicts.length;

  return (
    <div className="card">
      <div className="card-title">
        <span>🛡️</span> Deterministic Validation Results ({totalConflicts})
      </div>
      {totalConflicts === 0 ? (
        <div className="empty-state" style={{ color: '#34d399' }}>
          ✓ All deterministic validation checks passed clean. Zero conflicts detected.
        </div>
      ) : (
        <>
          {primaryConflicts.map((c, i) => (
            <div key={`p-${i}`} className="item-card" style={{ borderColor: '#ef4444' }}>
              <div className="item-header">
                <span className="item-title" style={{ color: '#f87171' }}>
                  {c.conflict_type}
                </span>
                <span className="badge badge-red">PRIMARY BLOCK</span>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#fca5a5', marginBottom: '0.2rem' }}>
                Reason Code: <code>{c.reason_code}</code>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>
                {c.message}
              </div>
            </div>
          ))}

          {secondaryConflicts.map((c, i) => (
            <div key={`s-${i}`} className="item-card" style={{ borderColor: '#f59e0b' }}>
              <div className="item-header">
                <span className="item-title" style={{ color: '#fbbf24' }}>
                  {c.conflict_type}
                </span>
                <span className="badge badge-amber">ESCALATION RULE</span>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#fcd34d', marginBottom: '0.2rem' }}>
                Reason Code: <code>{c.reason_code}</code>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>
                {c.message}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
};
