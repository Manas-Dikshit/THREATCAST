# THREATCAST Security

Defensive-security components (Module 5): MITRE ATT&CK mapping and explainability.

- `attack_mapping/` — deterministic stage→ATT&CK technique mapping tables (YAML, later phase). Every mapping records `source: "derived"` unless verified as dataset ground truth.
- `explainability/` — feature-attribution outputs conforming to the explanation contract (CONTRACT.md §12).

## Scope statement

THREATCAST is defensive cybersecurity software: traffic analysis, anomaly detection, attack prediction, monitoring, and defender decision support only. No exploit execution, malware deployment, credential theft, persistence mechanisms, or offensive automation.
