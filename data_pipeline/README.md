# THREATCAST Data Pipeline

Module 1: ingestion, parsing, feature extraction, time windows, preprocessing.

**Phase 1 status:** skeleton + package boundaries only.

## Scope

| Package | Responsibility |
|---------|----------------|
| `src/ingestion/` | PCAP, NetFlow/IPFIX, CSV flow records, auth logs intake |
| `src/parsers/` | format-specific readers normalising to flow records |
| `src/features/` | flow-level + packet-level feature extraction |
| `src/windows/` | time window engine → canonical `NetworkState` per `TIME_WINDOW_SECONDS` |
| `src/preprocessing/` | cleaning: missing/duplicate/infinite values, categoricals |
| `src/schemas/` | internal record types (canonical output = `backend/app/schemas.NetworkState`) |

## Hard rules

- Output states conform to the canonical NetworkState contract (CONTRACT.md §5); `features` is an open dict — additive extension only.
- CIC-IDS2018 columns are profiled from real files; never hard-coded from documentation.
- Streaming-ready seam: treat input as timestamped record streams so a live source can replace file readers later without changing downstream code.

See [../docs/DATA_SCHEMA.md](../docs/DATA_SCHEMA.md) for the full data schema.
