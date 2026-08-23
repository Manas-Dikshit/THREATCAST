"""Shared fixtures: small synthetic telemetry — no real dataset needed."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)


GENERIC_CSV = """timestamp,src_ip,src_port,dst_ip,dst_port,protocol,total_bytes,total_packets,flags,label
2026-01-01 10:00:01,10.0.0.1,443,10.0.0.9,5000,tcp,1200,10,S,ATTACK_A
2026-01-01 10:00:02,10.0.0.1,443,10.0.0.9,5001,tcp,800,8,SA,BENIGN
2026-01-01 10:00:03,10.0.0.2,80,10.0.0.9,5010,tcp,300,3,A,
2026-01-01 10:00:11,10.0.0.3,1234,10.0.0.10,53,udp,100,2,,BENIGN
2026-01-01 10:00:15,10.0.0.3,1235,10.0.0.10,53,udp,150,2,,BENIGN
2026-01-01 10:00:22,10.0.0.4,22,10.0.0.11,52000,tcp,2000,20,FA,ATTACK_B
"""

# CIC-IDS2018-flavoured file: BOM, spaced headers, microsecond durations,
# infinity values, duplicate row, NaN cells, mixed-case noise.
CIC_CSV = (
    "\ufeffDst Port,Flow Duration,Flow Bytes/s,Flow Pkts/s,Tot Fwd Pkts,Tot Bwd Pkts,"
    "TotLen Fwd Pkts,TotLen Bwd Pkts,Flow IAT Mean,Flow IAT Std,Flow IAT Max,"
    "SYN Flag Cnt,ACK Flag Cnt,RST Flag Cnt,Protocol,Src IP,Dst IP,Src Port,Timestamp,Label\n"
    "80,5000000,240.5,2.4,5,4,600,400,1000000,500000,2500000,1,1,0,6,10.0.0.1,10.0.0.99,40212,"
    '"14/02/2018 08:31:20",BENIGN\n'
    "445,12000000,Infinity,Infinity,12,10,1500,Infinity,1000000,200000,3000000,1,1,0,6,10.0.0.2,10.0.0.99,40213,"
    '"14/02/2018 08:31:30",SSH-Bruteforce\n'
    ",7000000,,,,,,, , , ,,,,,,,,\n"  # malformed row: no timestamp
    "53,3000000,100.0,1.0,2,2,200,180,1500000,100000,2900000,0,1,0,17,10.0.0.3,10.0.0.98,53111,"
    '"14/02/2018 08:31:40",BENIGN\n'
    "445,12000000,Infinity,Infinity,12,10,1500,Infinity,1000000,200000,3000000,1,1,0,6,10.0.0.2,10.0.0.99,40213,"
    '"14/02/2018 08:31:30",SSH-Bruteforce\n'
)


@pytest.fixture
def generic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "generic_flows.csv"
    path.write_text(GENERIC_CSV, encoding="utf-8")
    return path


@pytest.fixture
def cic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "cic_like.csv"
    path.write_text(CIC_CSV, encoding="utf-8")
    return path


@pytest.fixture
def empty_csv(tmp_path: Path) -> Path:
    path = tmp_path / "empty.csv"
    path.write_text(
        "timestamp,src_ip,src_port,dst_ip,dst_port,protocol,total_bytes,total_packets,label\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def garbage_csv(tmp_path: Path) -> Path:
    path = tmp_path / "garbage.csv"
    path.write_text("foo,bar\n1,2\n3,4\n", encoding="utf-8")
    return path


@pytest.fixture
def sample_records():
    from datetime import datetime, timedelta, timezone

    from data_pipeline.src.schemas.records import FlowRecord

    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    def rec(offset_s: float, **kw):
        defaults = dict(
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=40000,
            dst_port=80,
            protocol="tcp",
            total_bytes=500.0,
            total_packets=5.0,
        )
        defaults.update(kw)
        return FlowRecord(timestamp=t0 + timedelta(seconds=offset_s), **defaults)

    return rec
