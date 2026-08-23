import { describe, expect, it } from 'vitest'
import type { NetworkState } from '../src/types/contracts'

// Foundation check only: the contract types compile and the open feature dict accepts extension.
describe('contract types', () => {
  it('accepts an extended feature dict without schema change', () => {
    const state: NetworkState = {
      state_id: 'state_000001',
      timestamp_start: '2026-01-01T10:00:00Z',
      timestamp_end: '2026-01-01T10:00:10Z',
      window_seconds: 10,
      features: { flow_count: 120, syn_ratio: 0.42, some_future_feature: 1.0 },
      flow_summary: { unique_source_hosts: 8, unique_destination_hosts: 12 },
      label: null,
      label_source: null,
    }
    expect(state.features['syn_ratio']).toBe(0.42)
    expect(state.label).toBeNull()
  })
})
