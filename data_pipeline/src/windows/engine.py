"""Time-window engine: flows -> NetworkStates (CONTRACT.md section 5)."""

import math
from collections import Counter
from datetime import datetime, timedelta, timezone

from ..features.aggregator import aggregate_window, majority_label
from ..schemas.network_state import NetworkState
from ..schemas.records import FlowRecord

_EPS = 1e-6


def build_states(
    records: list[FlowRecord],
    *,
    window_seconds: float = 10.0,
    source_name: str | None = None,
) -> list[NetworkState]:
    """Bucket flow records into fixed windows anchored at epoch grid.

    Windows with zero records are skipped; state ids are sequential over the
    output (state_000001, ...). Labels: majority label of the window's flows,
    kept only when a verified source name is provided (ground truth rule).
    """
    if not records:
        return []
    ordered = sorted(records, key=lambda r: r.timestamp)
    anchor = math.floor(ordered[0].timestamp.timestamp() / window_seconds) * window_seconds

    buckets: dict[int, list[FlowRecord]] = {}
    for record in ordered:
        k = int((record.timestamp.timestamp() - anchor) // window_seconds)
        buckets.setdefault(k, []).append(record)

    states: list[NetworkState] = []
    for seqno, k in enumerate(sorted(buckets), start=1):
        group = buckets[k]
        features, summary = aggregate_window(group)
        label, _support = majority_label(group)
        start = datetime.fromtimestamp(anchor + k * window_seconds, tz=timezone.utc)
        states.append(
            NetworkState(
                state_id=f"state_{seqno:06d}",
                timestamp_start=start,
                timestamp_end=start + timedelta(seconds=window_seconds),
                window_seconds=window_seconds,
                features=features,
                flow_summary=summary,
                label=label if (label and source_name) else None,
                label_source=source_name if (label and source_name) else None,
            )
        )
    return states


def contiguous_runs(states: list[NetworkState], *, window_seconds: float | None = None) -> list[list[NetworkState]]:
    """Split chronologically sorted states into runs of adjacent windows.

    A gap (missing/empty window) breaks the run so sequences never span gaps —
    this is part of temporal-leakage prevention.
    """
    if not states:
        return []
    win = window_seconds or states[0].window_seconds
    runs: list[list[NetworkState]] = [[states[0]]]
    for prev, cur in zip(states, states[1:]):
        expected = prev.timestamp_start.timestamp() + prev.window_seconds
        if abs(cur.timestamp_start.timestamp() - expected) <= _EPS:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return runs


__all__ = ["build_states", "contiguous_runs"]
