import React, { useState } from 'react';
import type { SupportRequestInput } from '../types/mosaic';

interface RequestFormProps {
  onSubmit: (input: SupportRequestInput) => void;
  isLoading: boolean;
}

const PRESETS = [
  {
    label: 'ALLOW: Valid Refund',
    tag: 'allow',
    customerId: 'cust_allow_01',
    message: 'Please refund payment pay_101.',
    facts: { payment_verified: true, identity_verified: true },
    flags: [],
  },
  {
    label: 'BLOCK: Unverified Refund',
    tag: 'block',
    customerId: 'cust_block_02',
    message: 'Please process a refund for pay_102 right now!',
    facts: { payment_verified: true, identity_verified: false },
    flags: [],
  },
  {
    label: 'ESCALATE: Suspicious Lock',
    tag: 'escalate',
    customerId: 'cust_esc_03',
    message: 'Please restore login access for my account acc_707.',
    facts: { account_status: 'locked' },
    flags: ['SUSPICIOUS_LOCATION_LOGIN'],
  },
  {
    label: 'ALLOW: Multi-Intent Request',
    tag: 'allow',
    customerId: 'cust_multi_04',
    message: 'Cancel my active subscription sub_202 and also refund pay_101.',
    facts: { subscription_active: true, payment_verified: true, identity_verified: true },
    flags: [],
  },
];

export const RequestForm: React.FC<RequestFormProps> = ({ onSubmit, isLoading }) => {
  const [customerId, setCustomerId] = useState('cust_allow_01');
  const [message, setMessage] = useState('Please refund payment pay_101.');
  const [factsJson, setFactsJson] = useState('{\n  "payment_verified": true,\n  "identity_verified": true\n}');
  const [flagsInput, setFlagsInput] = useState('');

  const handleSelectPreset = (preset: typeof PRESETS[0]) => {
    setCustomerId(preset.customerId);
    setMessage(preset.message);
    setFactsJson(JSON.stringify(preset.facts, null, 2));
    setFlagsInput(preset.flags.join(', '));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    let initialFacts = {};
    try {
      if (factsJson.trim()) {
        initialFacts = JSON.parse(factsJson);
      }
    } catch {
      alert('Invalid JSON in State Facts field.');
      return;
    }

    const activeFlags = flagsInput
      .split(',')
      .map((f) => f.trim())
      .filter(Boolean);

    onSubmit({
      customer_id: customerId,
      raw_message: message,
      initial_facts: initialFacts,
      active_flags: activeFlags,
    });
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>💬</span> Support Request Intake
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <div className="form-label" style={{ marginBottom: '0.5rem' }}>Demo Test Scenarios</div>
        <div className="preset-buttons">
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              className="preset-btn"
              onClick={() => handleSelectPreset(p)}
            >
              <span>{p.label}</span>
              <span className={`preset-tag ${p.tag}`}>{p.tag}</span>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label">Customer ID</label>
          <input
            type="text"
            className="form-input"
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label className="form-label">Customer Message</label>
          <textarea
            className="form-textarea"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label className="form-label">Initial State Facts (JSON)</label>
          <textarea
            className="form-textarea"
            style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}
            value={factsJson}
            onChange={(e) => setFactsJson(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Active Flags (comma-separated)</label>
          <input
            type="text"
            className="form-input"
            placeholder="SUSPICIOUS_LOCATION_LOGIN, RISK_ACCOUNT"
            value={flagsInput}
            onChange={(e) => setFlagsInput(e.target.value)}
          />
        </div>

        <button type="submit" className="btn-submit" disabled={isLoading}>
          {isLoading ? 'Processing via MOSAIC Pipeline...' : 'Submit Request to MOSAIC →'}
        </button>
      </form>
    </div>
  );
};
