export interface Intent {
  id: string;
  category: string;
  intent_name: string;
  confidence: number;
  verbatim_text: string;
}

export interface ProposedAction {
  id: string;
  action_type: string;
  target_entity_id: string;
  agent_name: string;
  risk_level: string;
  parameters: Record<string, any>;
}

export interface ConflictRecord {
  conflict_type: string;
  reason_code: string;
  message: string;
  action_ids?: string[];
  rule_id?: string;
  field?: string;
}

export interface ActionExecutionRecord {
  action_id: string;
  action_type: string;
  target_entity_id: string;
  agent_name: string;
  status: 'SUCCESS' | 'BLOCKED' | 'ESCALATED' | 'FAILED' | 'SKIPPED';
  executed_at: string;
  result_details: Record<string, any>;
  message?: string;
}

export interface GovernanceExecutionResponse {
  request_id: string;
  customer_id: string;
  raw_message: string;
  detected_intents: Intent[];
  proposed_actions: ProposedAction[];
  current_state: Record<string, any>;
  primary_conflicts: ConflictRecord[];
  secondary_conflicts: ConflictRecord[];
  final_verdict: 'ALLOW' | 'BLOCK' | 'ESCALATED';
  human_readable_explanation: string;
  execution_status: 'EXECUTED' | 'BLOCKED' | 'ESCALATED' | 'FAILED' | 'SKIPPED';
  action_execution_records: ActionExecutionRecord[];
  processed_at: string;
}

export interface SupportRequestInput {
  customer_id: string;
  raw_message: string;
  initial_facts?: Record<string, any>;
  active_flags?: string[];
}
