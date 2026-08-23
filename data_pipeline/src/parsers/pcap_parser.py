"""PCAP parsing via Scapy.

Packets are grouped into bidirectional flows ((src,sport),(dst,dport),proto,
direction-normalized) and enriched with packet-level features other sources
cannot provide: TTL stats, TCP window, fragmentation, payload sizes,
retransmissions. Unavailable values stay None — nothing is fabricated.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..schemas.metadata import DatasetProfile, LabelKind
from ..schemas.records import FlowRecord

_FLAG_LETTERS = "FSRPAU"  # canonical output order


class _FlowState:
    __slots__ = (
        "start", "end", "packets", "nbytes", "flags", "ttls", "windows",
        "frag_count", "payloads", "seq_seen", "retrans",
    )

    def __init__(self) -> None:
        self.start: float | None = None
        self.end: float | None = None
        self.packets = 0
        self.nbytes = 0
        self.flags: set[str] = set()
        self.ttls: list[int] = []
        self.windows: list[int] = []
        self.frag_count = 0
        self.payloads: list[int] = []
        self.seq_seen: set[tuple[bool, int]] = set()
        self.retrans = 0


def parse_pcap(path: str | Path) -> tuple[list[FlowRecord], DatasetProfile]:
    from scapy.utils import PcapReader, PcapNgReader

    path = Path(path)
    reader_cls = PcapNgReader if _is_pcapng(path) else PcapReader

    flows: dict[tuple, _FlowState] = {}
    total_packets = 0
    skipped = 0

    with reader_cls(str(path)) as reader:
        for packet in reader:
            total_packets += 1
            ip_layer = packet.getlayer("IP") or packet.getlayer("IPv6")
            if ip_layer is None:
                skipped += 1
                continue
            src, dst = str(ip_layer.src), str(ip_layer.dst)
            tcp = packet.getlayer("TCP")
            udp = packet.getlayer("UDP")
            if tcp is not None:
                proto, sport, dport = "tcp", int(tcp.sport), int(tcp.dport)
            elif udp is not None:
                proto, sport, dport = "udp", int(udp.sport), int(udp.dport)
            else:
                proto, sport, dport = normalize_protocol(getattr(ip_layer, "nh", None)) or "other", 0, 0

            forward = (src, sport) <= (dst, dport)
            endpoints = ((src, sport), (dst, dport)) if forward else ((dst, dport), (src, sport))
            key = (*endpoints, proto)
            state = flows.setdefault(key, _FlowState())

            ts = float(packet.time)
            state.start = ts if state.start is None else min(state.start, ts)
            state.end = ts if state.end is None else max(state.end, ts)
            state.packets += 1
            state.nbytes += len(packet)

            if tcp is not None:
                value = int(tcp.flags)
                state.flags |= {letter for letter, bit in zip("FSRPAPU", (0x01, 0x02, 0x04, 0x08, 0x10, 0x20)) if value & bit}
                state.windows.append(int(tcp.window))
                seq_key = (forward, int(tcp.seq))
                if seq_key in state.seq_seen:
                    state.retrans += 1
                else:
                    state.seq_seen.add(seq_key)
                payload = bytes(tcp.payload)
                if payload:
                    state.payloads.append(len(payload))
            elif udp is not None:
                payload = bytes(udp.payload)
                if payload:
                    state.payloads.append(len(payload))

            ttl = getattr(ip_layer, "ttl", getattr(ip_layer, "hlim", None))
            if ttl is not None:
                state.ttls.append(int(ttl))
            try:
                if int(ip_layer.flags) & 0x2 or int(getattr(ip_layer, "frag", 0) or 0) > 0:
                    state.frag_count += 1
            except Exception:
                pass

    records: list[FlowRecord] = []
    for key, st in sorted(flows.items(), key=lambda kv: kv[1].start or 0.0):
        (a_ip, a_port), (b_ip, b_port), proto = key
        records.append(
            FlowRecord(
                timestamp=_epoch_to_datetime(st.start),
                src_ip=a_ip,
                dst_ip=b_ip,
                src_port=a_port or None,
                dst_port=b_port or None,
                protocol=proto,
                total_bytes=float(st.nbytes),
                total_packets=float(st.packets),
                duration_s=max((st.end or st.start) - st.start, 0.0),
                flags="".join(c for c in _FLAG_LETTERS + "PU"[:0] if c in st.flags) or None,
                ttl_mean=_mean(st.ttls),
                ttl_var=_var(st.ttls),
                tcp_window_mean=_mean(st.windows),
                retransmission_count=st.retrans or None,
                fragmentation_count=st.frag_count or None,
                payload_mean=_mean(st.payloads),
                payload_max=max(st.payloads) if st.payloads else None,
                label_kind=LabelKind.UNKNOWN,
            )
        )

    profile = DatasetProfile(
        source_file=str(path),
        detected_format="pcap",
        row_count=total_packets,
        column_count=0,
        warnings=[f"{skipped} non-IP packets skipped."] if skipped else [],
    )
    return records, profile


def _is_pcapng(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(4) == b"\x0a\x0d\x0d\x0a"


def _epoch_to_datetime(epoch_seconds: float) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


def _mean(values: list) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def _var(values: list) -> float | None:
    return round(float(np.var(values)), 6) if values else None
