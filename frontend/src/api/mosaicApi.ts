import type { GovernanceExecutionResponse, SupportRequestInput } from '../types/mosaic';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function submitSupportRequest(input: SupportRequestInput): Promise<GovernanceExecutionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/support/requests`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Request failed (${response.status}): ${errorText}`);
  }

  return response.json();
}
