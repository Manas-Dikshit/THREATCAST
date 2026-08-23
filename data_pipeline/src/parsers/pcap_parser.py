"""PCAP parsing via Scapy.

Packets are grouped into bidirectional flows (5-tuple, direction-normalized)
and enriched with packet-level features other sources cannot provide:
TTL stats, TCP window, fragmentation, payload sizes, retransmissions.
Unavailable values stay None — nothing is fabricated.
"""

from pathlib import Path

import numpy as np

from ..schemas.metadata import DatasetProfile, LabelKind
from ..schemas.records import FlowRecord
from .columns import normalize_protocol

_FLAG_BITS = {
    "F": 0x01,
    "S": 0x02,
    "R": 0x04,
    "P": 0x08,
    "A": 0x10,
    "U": 0x20,
}


def _flags_to_letters(value: int) -> str:
    return "".join(letter for letter, bit in _FLAG_BITS.items() if value & bit) or None


class _FlowState:
    __slots__ = (
        "start", "end", "packets", "bytes", "flags", "ttls", "windows",
        "frag_count", "payloads", "seq_seen", "retrans",
    )

    def __init__(self) -> None:
        self.start = None
        self.end = None
        self.packets = 0
        self.bytes = 0
        self.flags: set[str] = set()
        self.ttls: list[int] = []
        self.windows: list[int] = []
        self.frag_count = 0
        self.payloads: list[int] = []
        self.seq_seen: set[tuple[bool, int]] = set()
        self.retrans = 0


def parse_pcap(path: str | Path, *, label: str | None = None) -> tuple[list[FlowRecord], DatasetProfile]:
    """Parse a pcap/pcapng file into per-flow FlowRecords."""
    from scapy.utils import PcapReader, PcapNgReader

    path = Path(path)
    flows: dict[tuple, _FlowState] = {}
    total_packets = 0
    skipped = 0

    reader_cls = PcapNgReader if path.suffix.lower() in (".pcapng", ".pcap") and _is_pcapng(path) else PcapReader
    with reader_cls(str(path)) as reader:
        for packet in reader:
            total_packets += 1
            ip_layer = packet.getlayer("IP") or packet.getlayer("IPv6")
            if ip_layer is None:
                skipped += 1
                continue
            transport = packet.getlayer("TCP") or packet.getlayer("UDP")
            proto = "tcp" if packet.haslayer("TCP") else "udp" if packet.haslayer("UDP") else str(ip_layer.nh)
            src, dst = str(ip_layer.src), str(ip_layer.dst)
            sport = int(transport.sport) if transport is not None else 0
            dport = int(transport.dport) if transport is not None else 0

            forward = (src, sport) <= (dst, dport)
            key = ((src, sport), (dst, dport)) if forward else ((dst, dport), (src, sport))
            state = flows.setdefault(key, _FlowState())

            ts = float(packet.time)
            state.start = ts if state.start is None else min(state.start, ts)
            state.end = ts if state.end is None else max(state.end, ts)
            state.packets += 1
            try:
                state.bytes += len(packet)
            except Exception:
                pass

            tcp = packet.getlayer("TCP")
            if tcp is not None:
                state.flags |= set(_flags_to_letters(int(tcp.flags)) or "")
                state.windows.append(int(tcp.window))
                direction_forward = ((src, sport) == key[0])
                seq_key = (direction_forward, int(tcp.seq))
                if seq_key in state.seq_seen:
                    state.retrans += 1
                else:
                    state.seq_seen.add(seq_key)

            udp = packet.getlayer("UDP")
            if udp is not None:
                state.payloads.append(len(bytes(udp.payload)))

            # TTL from the network layer; fragmentation flags likewise.
            try:
                state.ttls.append(int(ip_layer.ttl))
            except AttributeError:
                pass
            try:
                if (ip_layer.flags & 0x2) or int(getattr(ip_layer, "frag", 0)) > 0:
                    state.frag_count += 1
            except Exception:
                pass
            payload = bytes(packet.payload)
            if payload:
                state.payloads.append(len(payload))

    records: list[FlowRecord] = []
    for key, st in sorted(flows.items(), key=lambda kv: kv[1].start or 0.0):
        (a_ip, a_port), (b_ip, b_port) = key
        forward_first = True
        records.append(
            FlowRecord(
                timestamp=np.datetime64(int(st.start * 1_000_000_000), "ns").astype(
                    "datetime64[ns]"
                ).astype(object) if False else _epoch_to_datetime(st.start),
                src_ip=a_ip if forward_first else a_ip,
                dst_ip=b_ip,
                src_port=a_port,
                dst_port=b_port,
                protocol=normalize_protocol(proto_of(key)),
                total_bytes=float(st.bytes),
                total_packets=float(st.packets),
                duration_s=max((st.end or st.start) - st.start, 0.0),
                flags="".join(sorted(st.flags, key="FSRPAPU".index)) if False else "".join(c for c in "FSRPAU" if c in st.flags) or None,
                ttl_mean=_mean(st.ttls),
                ttl_var=_var(st.ttls),
                tcp_window_mean=_mean(st.windows),
                retransmission_count=st.retrans or None,
                fragmentation_count=st.frag_count or None,
                payload_mean=_mean(st.payloads),
                payload_max=max(st.payloads) if st.payloads else None,
                label=label,
                label_kind=LabelKind.GROUND_TRUTH if label else LabelKind.UNKNOWN,
            )
        )

    profile = DatasetProfile(
        source_file=str(path),
        detected_format="pcap",
        row_count=total_packets,
        column_count=0,
        warnings=(
            [f"{skipped} non-IP packets skipped."]
            if skipped
            else []
        ),
    )
    return records, profile


def proto_of(key) -> str:
    return getattr(key, "__proto__", "") or ""


def _is_pcapng(path: Path) -> bool:
    with open(path, "rb") as fh:
        magic = fh.read(4)
    return magic == b"\x0a\x0d\x0d\x0a"  # pcapng section header block


def _epoch_to_datetime(epoch_seconds: float):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


def _mean(values: list) -> float | None:
    return float(np.mean(values)) if values else None


def _var(values: list) -> float | None:
    return float(np.var(values)) if values else None
