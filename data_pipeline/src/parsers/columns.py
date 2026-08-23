"""Header normalization and dynamic column mapping.

Datasets (CIC-IDS2018 especially) have inconsistent header names: BOM prefixes,
mixed case, spaces vs underscores, synonyms. We never assume a fixed schema:
headers are normalized and matched against alias sets; anything unmatched is
reported as unmapped instead of guessed.
"""

import re

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

# canonical field -> accepted normalized header names
ALIASES: dict[str, set[str]] = {
    "timestamp": {"timestamp", "flowtimestamp", "date", "datetime", "time", "ts", "stime", "starttime"},
    "src_ip": {"srcip", "sourceip", "srcaddr", "sa", "ipsrc"},
    "dst_ip": {"dstip", "destip", "destinationip", "dstaddr", "da", "ipdst"},
    "src_port": {"srcport", "sourceport", "sport", "sp"},
    "dst_port": {"dstport", "dport", "destinationport", "destport", "dp"},
    "protocol": {"protocol", "proto", "pr"},
    "total_bytes": {"totbytes", "totalbytes", "ibyt", "bytes", "flowbytes"},
    "total_packets": {"totpkts", "totalpackets", "totpkts", "ipkt", "pkts", "packets"},
    "duration": {"duration", "flowduration", "dur", "sessiontime"},
    "fwd_bytes": {"totlenfwdpkts", "fwdbytes", "subflowfwdbytes"},
    "bwd_bytes": {"totlenbwdpkts", "bwdbytes", "subflowbwdbytes"},
    "fwd_packets": {"totfwdpkts", "totalfwdpackets", "fwdpackets"},
    "bwd_packets": {"totbwdpkts", "totalbackwardpackets", "bwdpackets", "backwardpackets"},
    "iat_mean": {"flowiatmean", "iatmean"},
    "iat_std": {"flowiatstd", "iatstd"},
    "iat_max": {"flowiatmax", "iatmax"},
    "flag_syn": {"synflagcnt", "synflagcount", "tcpsynflagcount"},
    "flag_ack": {"ackflagcnt", "ackflagcount", "tcpackflagcount"},
    "flag_fin": {"finflagcnt", "finflagcount", "tcpfinflagcount"},
    "flag_rst": {"rstflagcnt", "rstflagcount", "tcprstflagcount"},
    "flag_psh": {"pshflagcnt", "pshflagcount", "tcppshflagcount"},
    "flag_urg": {"urgflagcnt", "urgflagcount", "tcpurgflagcount"},
    "label": {"label", "attack", "class", "category", "attacktype"},
    "flags": {"flags", "tcpflags", "tcpflag", "flowflags", "flagstring"},
}

# CIC-IDS2018 duration/IAT columns are expressed in microseconds.
MICROSECOND_COLUMNS = {"flowduration", "flowiatmean", "flowiatstd", "flowiatmax"}

_PROTOCOL_NAMES = {1: "icmp", 6: "tcp", 17: "udp", 47: "gre", 58: "icmpv6"}


def normalize_header(name: str) -> str:
    """'ï»¿ Dst Port ' -> 'dstport'"""
    return _NORMALIZE_RE.sub("", str(name).strip().lower().lstrip("\ufeff"))


def map_columns(columns) -> dict[str, str]:
    """Return {canonical_field: original_header} for recognized columns."""
    mapping: dict[str, str] = {}
    for original in columns:
        norm = normalize_header(original)
        for canonical, aliases in ALIASES.items():
            if norm in aliases and canonical not in mapping:
                mapping[canonical] = original
                break
    return mapping


def normalize_protocol(value) -> str | None:
    """Map protocol numbers to lowercase names; keep unknown values verbatim."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        num = int(float(str(value).strip()))
    except (ValueError, TypeError):
        text = str(value).strip().lower()
        return text or None
    return _PROTOCOL_NAMES.get(num, str(num))
