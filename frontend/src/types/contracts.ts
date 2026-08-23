// TypeScript mirror of the canonical contracts (CONTRACT.md §5-§7).
// Keep in sync with backend/app/schemas.

export interface FlowSummary {
  unique_source_hosts: number
  unique_destination_hosts: number
}

export interface NetworkState {
  state_id: string
  timestamp_start: string
  timestamp_end: string
  window_seconds: number
  features: Record<string, number> // open dict — additive extension only
  flow_summary: FlowSummary
  label: string | null
  label_source: string | null
}

export interface NetworkStateSequence {
  sequence_id: string
  states: NetworkState[]
  sequence_length: number
  window_seconds: number
  target_state: NetworkState | null
}

export interface PredictedStage {
  id: string | null
  name: string | null
  confidence: number | null
  source: string | null
}

export interface FeatureContribution {
  feature: string
  contribution: number
}

export interface PredictionResult {
  prediction_id: string
  timestamp: string
  risk_score: number
  malicious_probability: number
  confidence: number
  predicted_stage: PredictedStage
  future_states: unknown[]
  feature_contributions: FeatureContribution[]
  model: { name: string; version: string }
}

export interface ApiError {
  error: {
    code: string
    message: string
    details: Record<string, unknown>
    request_id: string
  }
}
