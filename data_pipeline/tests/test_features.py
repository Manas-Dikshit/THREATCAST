"""Feature aggregation tests: honest computation, omission of unavailable data."""

import pytest

from data_pipeline.src.features.aggregator import aggregate_window, majority_label


def test_basic_counts(sample_records):
    recs = [
        sample_records(0, total_bytes=100, total_packets=4),
        sample_records(1, total_bytes=50, total_packets=2),
    ]
    features, summary = aggregate_window(recs)
    assert features["flow_count"] == 2
    assert features["packet_count"] == 6
    assert features["byte_count"] == 150
    assert summary.unique_source_hosts == 1
    assert summary.unique_destination_hosts == 1


def test_flag_ratios(sample_records):
    recs = [
        sample_records(0, flags="S"),
        sample_records(1, flags="SA"),
        sample_records(2, flags="SA"),
    ]
    features, _ = aggregate_window(recs)
    assert features["syn_ratio"] == pytest.approx(1.0)
    assert features["ack_ratio"] == pytest.approx(2 / 3)
    assert "rst_ratio" in features and features["rst_ratio"] == 0.0


def test_unavailable_features_omitted_not_fabricated(sample_records):
    """CSV-derived records carry no TTL/window/retransmission data -> keys absent."""
    features, _ = aggregate_window([sample_records(0)])
    for absent in ("ttl_mean", "ttl_variance", "tcp_window_mean",
                   "retransmission_count", "fragmentation_count",
                   "payload_size_mean", "payload_size_max"):
        assert absent not in features


def test_pcap_level_features_aggregated(sample_records):
    recs = [
        sample_records(0, ttl_mean=64.0, ttl_var=0.5, tcp_window_mean=64240,
                       retransmission_count=2, fragmentation_count=1,
                       payload_mean=300.0, payload_max=500.0, total_packets=10),
        sample_records(1, ttl_mean=60.0, tcp_window_mean=1000, total_packets=2),
    ]
    features, _ = aggregate_window(recs)
    assert features["ttl_mean"] == pytest.approx(62.0)      # packet-weighted
    assert features["ttl_variance"] == pytest.approx(0.5)
    assert features["retransmission_count"] == 2
    assert features["fragmentation_count"] == 1
    assert features["payload_size_max"] == 500


def test_iat_from_flow_stats_preferred(sample_records):
    recs = [
        sample_records(0, iat_mean_s=0.01, iat_var_s=0.0001, iat_max_s=0.05),
        sample_records(1, iat_mean_s=0.03, iat_var_s=0.0003, iat_max_s=0.09),
    ]
    features, _ = aggregate_window(recs)
    assert features["mean_iat"] == pytest.approx(0.02)
    assert features["iat_variance"] == pytest.approx(0.0002)
    assert features["iat_max"] == pytest.approx(0.09)


def test_iat_fallback_inter_flow_gaps(sample_records):
    """No per-flow IAT stats: gaps between flow arrivals are measured instead."""
    recs = [sample_records(0), sample_records(2)]   # 2 s apart
    features, _ = aggregate_window(recs)
    assert features["mean_iat"] == pytest.approx(2.0)


def test_bidirectional_byte_ratio(sample_records):
    recs = [sample_records(0, fwd_bytes=750, bwd_bytes=250)]
    features, _ = aggregate_window(recs)
    assert features["fwd_byte_ratio"] == pytest.approx(0.75)


def test_port_scan_score_heuristic(sample_records):
    scanned = [sample_records(i * 0.1, dst_port=100 + i) for i in range(10)]
    features, _ = aggregate_window(scanned)
    assert features["port_scan_score"] == pytest.approx(1.0)  # one host, 10 distinct ports

    normal = [
        sample_records(i, src_ip=f"10.0.0.{i}", dst_port=80)
        for i in range(10)
    ]
    features_normal, _ = aggregate_window(normal)
    assert features_normal["port_scan_score"] == 0.0

    quiet = [sample_records(0), sample_records(1)]            # below min-flow threshold
    features_quiet, _ = aggregate_window(quiet)
    assert features_quiet["port_scan_score"] == 0.0


def test_unique_ports_and_hosts(sample_records):
    recs = [
        sample_records(0, src_ip="10.0.0.1", src_port=1000, dst_ip="10.0.0.9", dst_port=80),
        sample_records(1, src_ip="10.0.0.2", src_port=2000, dst_ip="10.0.0.9", dst_port=443),
    ]
    features, summary = aggregate_window(recs)
    assert features["unique_src_ports"] == 2
    assert features["unique_dst_ports"] == 2
    assert summary.unique_source_hosts == 2
    assert summary.unique_destination_hosts == 1


def test_majority_label_tie_deterministic(sample_records):
    recs = [
        sample_records(0, label="B"),
        sample_records(1, label="A"),
        sample_records(2, label=None),
    ]
    label, support = majority_label(recs)
    assert (label, support) == ("A", 1)     # tie B/A -> alphabetical
    assert majority_label([sample_records(0)]) == (None, 0)
