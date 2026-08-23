# THREATCAST Integration

Cross-module integration tests, fixtures and scripts (Module 5).

- `tests/` — end-to-end contract tests once modules exist (later phases): pipeline → states → predict API round-trips.
- `fixtures/` — small hand-crafted sample inputs (tiny CSV flow records etc.). Real datasets are never committed.
- `scripts/` — cross-module helper scripts.

Phase 1 provides the structure; foundation validation lives in `../tests/smoke`.
