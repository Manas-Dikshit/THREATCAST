"""Window-level feature aggregation over normalized FlowRecords.

Rules:
- Every feature is computed from what the records actually carry.
- Features whose inputs are unavailable are OMITTED from the dict entirely
  (never zero-filled, never fabricated).
- All outputs are floats rounded to 6 decimals for deterministic serialization.
"""

from collections import Counter

import numpy as np

from ..schemas.network_state import FlowSummary
from ..schemas.records import FlowRecord

_FLAG_RATIO_KEYS = {
    "S": "syn_ratio",
    "A": "ack_ratio",
    "F": "fin_ratio",
    "R": "rst_ratio",
    "P": "psh_ratio",
    "U": "urg_ratio",
}


def _r6(value: float) -> float:
    return round(float(value), 6)


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    total_w = sum(weights)
    if total_w <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def aggregate_window(
    records: list[FlowRecord],
    *,
    min_flows_for_scan_score: int = 5,
) -> tuple[dict[str, float], FlowSummary]:
    """Aggregate one window's flows into canonical feature dict + flow summary."""
    features: dict[str, float] = {}
    n = len(records)

    features["flow_count"] = float(n)

    packet_counts = [(r.total_packets, r.total_packets or 0.0) for r in records if r.total_packets is not None]
    if packet_counts:
        features["packet_count"] = _r6(sum(w for _, w in packet_counts))
    byte_values = [(r.total_bytes, r.total_packets or 1.0) for r in records if r.total_bytes is not None]
    if byte_values:
        features["byte_count"] = _r6(sum(v for v, _ in byte_values))

    # TCP flag ratios over flows whose flags are known.
    known_flags = [r.flag_letters() for r in records if r.flags is not None]
    if known_flags:
        for letter, key in _FLAG_RATIO_KEYS.items():
            features[key] = _r6(sum(letter in fl for fl in known_flags) / len(known_flags))

    # Inter-arrival time statistics: prefer per-flow IAT stats (CIC columns /
    # pcap aggregation); fall back to inter-flow arrival gaps in this window.
    iat_means = [r.iat_mean_s for r in records if r.iat_mean_s is not None]
    if iat_means:
        features["mean_iat"] = _r6(np.mean(iat_means))
        iat_vars = [r.iat_var_s for r in records if r.iat_var_s is not None]
        if iat_vars:
            # Approximation: mean of per-flow variances (per-record raw samples
            # are not retained by CSV sources).
            features["iat_variance"] = _r6(np.mean(iat_vars))
        iat_maxes = [r.iat_max_s for r in records if r.iat_max_s is not None]
        if iat_maxes:
            features["iat_max"] = _r6(max(iat_maxes))
    elif n >= 2:
        times = sorted(r.timestamp.timestamp() for r in records)
        gaps = np.diff(times)
        features["mean_iat"] = _r6(gaps.mean())
        features["iat_variance"] = _r6(gaps.var())

    # Bidirectional byte balance when directional counters exist.
    fwd_total = sum(r.fwd_bytes for r in records if r.fwd_bytes is not None)
    bwd_total = sum(r.bwd_bytes for r in records if r.bwd_bytes is not None)
    if fwd_total + bwd_total > 0:
        features["fwd_byte_ratio"] = _r6(fwd_total / (fwd_total + bwd_total))

    # Port/host diversity.
    src_ports = {r.src_port for r in records if r.src_port is not None}
    dst_ports = {r.dst_port for r in records if r.dst_port is not None}
    if src_ports:
        features["unique_src_ports"] = float(len(src_ports))
    if dst_ports:
        features["unique_dst_ports"] = float(len(dst_ports))

    # Port-scan indicator: max share of distinct destination ports contacted
    # by a single source host (only meaningful once enough flows exist).
    by_src: dict[str, list[FlowRecord]] = {}
    for r in records:
        if r.src_ip:
            by_src.setdefault(r.src_ip, []).append(r)
    scan_scores = [
        len({x.dst_port for x in group if x.dst_port is not None}) / len(group)
        for group in by_src.values()
        if len(group) >= min_flows_for_scan_score
    ]
    features["port_scan_score"] = _r6(max(scan_scores)) if scan_scores else 0.0

    # Packet-level aggregates (PCAP path).
    ttl_means = [(r.ttl_mean, r.total_packets or 1.0) for r in records if r.ttl_mean is not None]
    if ttl_means:
        features["ttl_mean"] = _r6(_weighted_mean([v for v, _ in ttl_means], [w for _, w in ttl_means]))
        ttl_vars = [r.ttl_var for r in records if r.ttl_var is not None]
        if ttl_vars:
            features["ttl_variance"] = _r6(float(np.mean(ttl_vars)))
    tcp_windows = [(r.tcp_window_mean, r.total_packets or 1.0) for r in records if r.tcp_window_mean is not None]
    if tcp_windows:
        features["tcp_window_mean"] = _r6(_weighted_mean([v for v, _ in tcp_windows], [w for _, w in tcp_windows]))
    retrans = sum(r.retransmission_count for r in records if r.retransmission_count)
    if retrans:
        features["retransmission_count"] = float(retrans)
    frags = sum(r.fragmentation_count for r in records if r.fragmentation_count)
    if frags:
        features["fragmentation_count"] = float(frags)
    payload_means = [r.payload_mean for r in records if r.payload_mean is not None]
    if payload_means:
        features["payload_size_mean"] = _r6(np.mean(payload_means))
    payload_maxes = [r.payload_max for r in records if r.payload_max is not None]
    if payload_maxes:
        features["payload_size_max"] = _r6(max(payload_maxes))

    src_ips = {r.src_ip for r in records if r.src_ip}
    dst_ips = {r.dst_ip for r in records if r.dst_ip}
    summary = FlowSummary(
        unique_source_hosts=len(src_ips),
        unique_destination_hosts=len(dst_ips),
    )
    return features, summary


def majority_label(records: list[FlowRecord]) -> tuple[str | None, int]:
    """Most common non-null label; ties resolved alphabetically (deterministic)."""
    labels = [r.label for r in records if r.label]
    if not labels:
        return None, 0
    counts = Counter(labels)
    top = max(counts.values())
    winners = sorted(label for label, c in counts.items() if c == top)
    return winners[0], top


__all__ = ["aggregate_window", "majority_label"]
