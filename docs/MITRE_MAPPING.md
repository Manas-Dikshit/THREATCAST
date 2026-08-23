# MITRE ATT&CK Mapping

Status: concept contract — implementation in a later phase within `security/attack_mapping/`.

## Stage taxonomy (derived, not ground truth)

| # | Derived stage | Typical signal pattern |
|---|---------------|------------------------|
| 1 | Reconnaissance | port-scan score high, many unique dst ports, low byte counts |
| 2 | Initial Access | spikes in inbound flows to service ports |
| 3 | Lateral Movement | many internal-to-internal flows (e.g. SMB/445) |
| 4 | Command and Control | periodic beaconing (regular IAT), external dst |
| 5 | Exfiltration | large outbound byte volumes from single hosts |

## Ground truth vs derived stage

- **GROUND TRUTH**: only labels verifiably present in a supplied dataset (e.g. CIC-IDS2018 attack categories). Must be confirmed against the actual downloaded files.
- **DERIVED STAGE**: anything inferred by THREATCAST models or mapping heuristics. Every API output carries `"source": "derived"` (or the verified source name) in `predicted_stage`.

**Never claim a derived stage is directly provided by CIC-IDS2018 unless verified against the real files.**

## Planned mechanism

1. Attack-predictor head outputs stage probabilities.
2. A deterministic YAML table in `security/attack_mapping/` maps (stage, top contributing features) → ATT&CK Enterprise technique IDs (e.g. T1046 Network Service Scanning).
3. Mapped techniques are attached to predictions/explanations; the dashboard renders them read-only for defenders.
