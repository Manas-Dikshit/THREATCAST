# THREATCAST Data Pipeline

Module 1: ingestion, parsing, feature extraction, preprocessing, time windows, ML dataset export.

**Status (Phase 2): fully implemented for CSV / CIC-IDS2018 / NetFlow-style CSV / PCAP inputs.**

## Supported inputs

| Format | Detection | Notes |
|--------|-----------|-------|
| Generic flow CSV | `.csv`, alias-mapped headers | any column subset; missing fields stay `null` |
| CIC-IDS2018 CSV | `.csv` + header sniffing | BOMs, spaced/mixed-case headers, µs units, Infinity, duplicates, NaN — all handled dynamically |
| NetFlow/IPFIX-style CSV exports | `.csv` | softflowd-style short aliases (`sa`,`da`,`sp`,`dp`,`pr`,`ipkt`,`ibyt`) |
| PCAP / PCAPNG | `.pcap`/`.pcapng` magic bytes | Scapy-based; packet-level features |

Binary NetFlow v5/v9 and IPFIX are **not** parsed (limitation); export to CSV first.

## Pipeline architecture

```
file ──> ingestion.loader.load_telemetry()
          ├─ parsers.csv_flows  (CSV path: raw read -> dynamic column mapping)
          ├─ parsers.pcap_parser (PCAP path: packet -> bidirectional flows)
          └─ preprocessing.cleaning (dupes, NaN/inf, malformed rows)
        ──> list[FlowRecord]                (normalized intermediate contract)
        ──> windows.engine.build_states()   (fixed TIME_WINDOW_SECONDS buckets)
        ──> list[NetworkState]              (canonical CONTRACT.md §5 objects)
        ──> windows.sequences.build_sequences()  (sliding S(t-L+1..t) + S(t+K))
        ──> list[NetworkStateSequence]      (canonical CONTRACT.md §6 objects)
        ──> windows.dataset.export_ml_dataset()  (Parquet + JSONL + NPZ + metadata)
```

Orchestrator: `data_pipeline.src.pipeline.run_pipeline()`.

## Feature definitions

Window-level features are computed from what records actually carry. **Unavailable features are omitted from the dict entirely — never zero-filled or fabricated.**

| Feature | Source | Definition |
|---------|--------|------------|
| `flow_count` | all | flows in window |
| `packet_count`, `byte_count` | CSV/PCAP | sums of per-flow totals (CIC: fwd+bwd components) |
| `syn_ratio` … `urg_ratio` | flags known | fraction of flag-bearing flows containing each letter |
| `mean_iat`, `iat_variance`, `iat_max` | preferred | mean/var/max of per-flow IAT stats |
| | fallback | inter-flow arrival gaps inside the window |
| `fwd_byte_ratio` | directional bytes | forward share of observed fwd+bwd bytes |
| `unique_src_ports`, `unique_dst_ports` | ports known | distinct port counts |
| `port_scan_score` | src IPs + dst ports | max over source hosts with ≥5 flows of distinct-dst-ports ÷ flows-from-host (heuristic indicator, derived) |
| `ttl_mean`, `ttl_variance` | PCAP only | packet-weighted TTL stats |
| `tcp_window_mean` | PCAP only | packet-weighted TCP window |
| `retransmission_count` | PCAP only | repeated TCP seq in same direction |
| `fragmentation_count` | PCAP only | IP MF-flag / frag-offset packets |
| `payload_size_mean`, `payload_size_max` | PCAP only | transport payload stats |

## State & sequence schema

Canonical objects live in `data_pipeline/src/schemas/network_state.py`, kept
schema-identical to `backend/app/schemas` (a compat test guards drift).
State ids are sequential (`state_000001`…); labels are the majority label of
the window's flows and are attached **only** when a verified `source_name` is
provided (`label_source`), per the ground-truth rule.

Sequences slide over *contiguous* state runs only — a gap (empty window) breaks
the run, so no sequence ever spans missing time. `target_state = S(t+K)` when it
exists inside the run, else `None`. Defaults: L=5, K=3, window=10 s (env-configurable).

## Preprocessing & leakage prevention

- `cleaning.clean_flow_table`: ±inf → null, exact duplicates dropped, unparseable-timestamp rows dropped, chronological sort. Everything reported.
- `profiler.profile_dataframe`: dynamic inspection report (`DatasetProfile`) — dtypes, null/inf counts, duplicates, label distribution, timestamp range, canonical column mapping. Saved as `dataset_profile.json`.
- `scaling.FeatureScaler`: z-score normalization **fitted on the train split only**; validation/test are transformed with frozen stats. Constant features get std=1 guard.
- `sequences.split_time_aware`: chronological train/val/test split by last-input timestamp; nothing is shuffled across the temporal boundary.

## Labels

Original dataset labels are preserved verbatim. `LabelKind` distinguishes:
`ground_truth` (verified source dataset), `derived` (THREATCAST inference),
`unknown`. No MITRE ATT&CK stage ground truth is invented here.

## Commands

```powershell
# End-to-end: file -> states -> sequences -> ML dataset artifacts
python -m data_pipeline.cli --input data\flows.csv --out out\ml_dataset --source-name cic_ids2018

# Options: --window 10  --seq-len 5  --horizon 3  --profile-only

# Programmatic
from data_pipeline import run_pipeline
result = run_pipeline("flows.pcap", "out/ml", source_name="lab_capture")

# Tests
pytest data_pipeline\tests -v
```

### Example output artifacts

| File | Content |
|------|---------|
| `states.parquet` | one row per NetworkState (features flattened) |
| `sequences.jsonl` | full NetworkStateSequence contract JSON, one per line |
| `tensors.npz` | `X [N,L,F] float32`, `Y [N,F] float32`, `target_mask [N]`, `label_ids [N]` |
| `feature_schema.json` | ordered feature names — the tensor column order |
| `preprocessing_metadata.json` | version, normalization stats, fit window, config |
| `label_mappings.json` | label → id (`-1` = unlabeled) |
| `dataset_profile.json` | raw-file profiling report |

## Limitations

- Binary NetFlow v5/v9 / IPFIX not supported (use CSV exports).
- Empty windows are skipped rather than emitted as zero-states; sequences break at gaps.
- Window IAT variance uses mean-of-per-flow-variances (raw packet samples aren't retained by CSV sources).
- `port_scan_score` is a derived heuristic indicator, not ground truth.
- Parquet requires `pyarrow`; everything else is stdlib/pandas.
