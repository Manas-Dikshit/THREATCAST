"""PCAP parser tests using a tiny synthetic capture built with Scapy."""

import pytest

scapy = pytest.importorskip("scapy")

from scapy.all import Ether, IP, TCP, UDP, wrpcap  # noqa: E402

from data_pipeline.src.parsers.pcap_parser import parse_pcap  # noqa: E402


@pytest.fixture
def tiny_pcap(tmp_path):
    base = 1767261600.0  # fixed epoch, keeps tests deterministic
    pkts = [
        # flow A: 10.0.0.1:4444 -> 10.0.0.2:80 — SYN handshake + one retransmission
        _tcp("10.0.0.1", 4444, "10.0.0.2", 80, flags="S", seq=100, t=base),
        _tcp("10.0.0.2", 80, "10.0.0.1", 4444, flags="SA", seq=500, t=base + 0.1),
        _tcp("10.0.0.1", 4444, "10.0.0.2", 80, flags="A", seq=101, t=base + 0.2),
        _tcp("10.0.0.1", 4444, "10.0.0.2", 80, flags="A", seq=101, t=base + 0.3),  # retrans
        # flow B: UDP DNS-style
        Ether(src="aa:bb:cc:dd:ee:01", dst="aa:bb:cc:dd:ee:02")
        / IP(src="10.0.0.3", dst="8.8.8.8", ttl=128)
        / UDP(sport=53000, dport=53)
        / (b"x" * 20),
    ]
    path = tmp_path / "tiny.pcap"
    wrpcap(str(path), pkts)
    return path


def _tcp(src, sport, dst, dport, *, flags, seq, t):
    pkt = (
        Ether(src="aa:bb:cc:dd:ee:01", dst="aa:bb:cc:dd:ee:02")
        / IP(src=src, dst=dst)
        / TCP(sport=sport, dport=dport, flags=flags, seq=seq)
    )
    pkt.time = t
    return pkt


def test_flow_grouping_and_direction_normalization(tiny_pcap):
    records, profile = parse_pcap(tiny_pcap)
    assert profile.detected_format == "pcap"
    assert len(records) == 2                      # one TCP flow + one UDP flow
    tcp = next(r for r in records if r.protocol == "tcp")
    udp = next(r for r in records if r.protocol == "udp")

    assert tcp.src_ip == "10.0.0.1" and tcp.dst_port == 80
    assert tcp.total_packets == 4
    assert set(tcp.flags) == {"S", "A"}           # union of observed flag letters
    assert tcp.retransmission_count == 1          # duplicated seq=100 packet


def test_packet_level_features_present_only_for_pcap(tiny_pcap):
    records, _ = parse_pcap(tiny_pcap)
    tcp = next(r for r in records if r.protocol == "tcp")
    assert tcp.ttl_mean == pytest.approx(64.0)    # scapy default TTL
    assert tcp.tcp_window_mean is not None and tcp.tcp_window_mean > 0
    assert tcp.duration_s >= 0.2
    udp = next(r for r in records if r.protocol == "udp")
    assert udp.payload_mean is not None           # payload captured from UDP data


def test_empty_pcap(tmp_path):
    path = tmp_path / "empty.pcap"
    wrpcap(str(path), [])
    records, profile = parse_pcap(path)
    assert records == []
    assert profile.row_count == 0
