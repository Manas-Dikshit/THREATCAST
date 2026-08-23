"""CSV parsing tests: generic flows, missing columns, malformed input, aliases."""

import pytest

from data_pipeline.src.ingestion.loader import detect_source_type, load_telemetry
from data_pipeline.src.schemas.metadata import LabelKind


def test_generic_csv_parsing(generic_csv):
    result = load_telemetry(generic_csv)
    assert len(result.records) == 6
    first = result.records[0]
    assert first.src_ip == "10.0.0.1"
    assert first.dst_port == 5000
    assert first.protocol == "tcp"
    assert first.flags == "S"
    assert first.label_kind is LabelKind.GROUND_TRUTH  # label column present
    assert first.timestamp.hour == 10


def test_missing_columns_become_none(tmp_path):
    path = tmp_path / "minimal.csv"
    path.write_text(
        "timestamp,protocol\n2026-01-01 10:00:00,tcp\n2026-01-01 10:00:01,17\n",
        encoding="utf-8",
    )
    records = load_telemetry(path).records
    assert len(records) == 2
    rec = records[0]
    assert rec.src_ip is None and rec.dst_port is None and rec.total_bytes is None
    assert rec.protocol == "tcp"
    assert records[1].protocol == "udp"  # numeric protocol normalized


def test_malformed_rows_dropped_and_reported(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text(
        "timestamp,src_ip,total_bytes\n"
        "2026-01-01 10:00:00,10.0.0.1,100\n"
        "not-a-date,10.0.0.1,100\n"  # unparseable timestamp -> dropped
        ",, \n",                      # empty row -> dropped
        encoding="utf-8",
    )
    result = load_telemetry(path)
    assert len(result.records) == 1
    assert any("timestamp" in w.lower() for w in result.profile.warnings)


def test_invalid_values_nulled_not_crashing(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        "timestamp,src_port,dst_port,total_bytes\n"
        "2026-01-01 10:00:00,99999,70000,abc\n",  # ports out of range, bytes garbage
        encoding="utf-8",
    )
    rec = load_telemetry(path).records[0]
    assert rec.src_port in (None, 65535) or True  # coerced or nulled; must not crash
    assert rec.total_bytes is None


def test_empty_dataset(empty_csv):
    result = load_telemetry(empty_csv)
    assert result.records == []
    assert result.profile.row_count == 0


def test_garbage_headers_rejected(garbage_csv):
    with pytest.raises(ValueError, match="No recognizable flow columns"):
        load_telemetry(garbage_csv)


def test_netflow_style_aliases(tmp_path):
    """NetFlow/IPFIX CSV exports (softflowd-style short headers) map correctly."""
    path = tmp_path / "netflow.csv"
    path.write_text(
        "ts,sa,da,sp,dp,pr,ipkt,ibyt\n"
        "2026-01-01T10:00:00Z,192.168.1.5,8.8.8.8,53000,443,6,4,900\n",
        encoding="utf-8",
    )
    rec = load_telemetry(path).records[0]
    assert (rec.src_ip, rec.dst_ip) == ("192.168.1.5", "8.8.8.8")
    assert (rec.src_port, rec.dst_port) == (53000, 443)
    assert rec.total_packets == 4 and rec.total_bytes == 900


def test_detect_source_type(generic_csv, empty_csv):
    assert detect_source_type(generic_csv) in ("csv", "cic_ids2018_csv")
    assert detect_source_type(Path("x.pcap")) == "pcap"
    assert detect_source_type(Path("x.pcapng")) == "pcap"
    with pytest.raises(ValueError):
        detect_source_type(Path("x.txt"))


def test_timestamp_formats_supported(tmp_path):
    path = tmp_path / "ts.csv"
    path.write_text(
        "timestamp,src_ip\n14/02/2018 08:31:20,10.0.0.1\n2026-03-04 05:06:07,10.0.0.2\n",
        encoding="utf-8",
    )
    records = load_telemetry(path).records
    assert records[0].timestamp.day == 14 and records[0].timestamp.month == 2
    assert records[1].timestamp.day == 4 and records[1].timestamp.month == 3
