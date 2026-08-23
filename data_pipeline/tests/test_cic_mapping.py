"""CIC-IDS2018-style parsing: dynamic column discovery, units, quirks."""

from data_pipeline.src.ingestion.loader import load_telemetry
from data_pipeline.src.preprocessing.profiler import save_profile
from data_pipeline.src.schemas.metadata import DatasetProfile, LabelKind


def test_cic_detection(cic_csv):
    result = load_telemetry(cic_csv)
    assert result.source_type == "cic_ids2018_csv"


def test_dynamic_column_mapping(cic_csv):
    profile = load_telemetry(cic_csv).profile
    assert set(profile.mapped_columns) >= {
        "dst_port", "duration", "fwd_packets", "bwd_packets",
        "fwd_bytes", "bwd_bytes", "iat_mean", "iat_std", "iat_max",
        "timestamp", "label", "protocol", "src_ip", "dst_ip",
    }
    assert "Flow Bytes/s" in profile.unmapped_columns  # unknown columns reported, not guessed
    assert profile.mapped_columns["dst_port"] == "Dst Port"


def test_microsecond_units_converted(cic_csv):
    rec = load_telemetry(cic_csv).records[0]
    assert rec.duration_s == 5.0          # 5000000 us -> 5 s
    assert rec.iat_mean_s == 1.0           # 1000000 us -> 1 s
    assert rec.total_bytes == 1000.0       # fwd 600 + bwd 400


def test_flags_from_count_columns(cic_csv):
    recs = load_telemetry(cic_csv).records
    assert recs[0].flags == "SA"           # SYN + ACK counts > 0, RST == 0
    assert recs[2].flags == "A"


def test_labels_preserved_as_ground_truth(cic_csv):
    result = load_telemetry(cic_csv)
    labels = {r.label for r in result.records}
    assert labels == {"BENIGN", "SSH-Bruteforce"}   # original strings verbatim
    assert all(r.label_kind is LabelKind.GROUND_TRUTH for r in result.records)
    assert result.profile.label_distribution["SSH-Bruteforce"] == 2  # raw file incl. dup


def test_duplicates_infinity_malformed_handled(cic_csv):
    result = load_telemetry(cic_csv)
    # 4 raw rows -> malformed dropped (no timestamp), duplicate removed -> 3 records
    assert len(result.records) == 3
    warnings = " | ".join(result.profile.warnings).lower()
    assert "duplicate" in warnings
    assert result.profile.row_count == 4   # raw row count before cleaning
    timestamps = sorted(r.timestamp.isoformat() for r in result.records)
    assert timestamps[0] < timestamps[-1]


def test_inf_value_nulled_in_mapped_column(cic_csv):
    """Infinity inside a mapped numeric column must be nulled, not propagated."""
    import math

    recs = load_telemetry(cic_csv).records
    ssh_row = [r for r in recs if r.label == "SSH-Bruteforce"][0]
    assert ssh_row.bwd_bytes is None                       # Infinity -> null
    assert ssh_row.total_bytes == 1500.0                   # sum over observed components
    values = [v for v in vars(ssh_row).values() if isinstance(v, float)]
    assert all(math.isfinite(v) for v in values)


def test_profile_report_round_trip(cic_csv, tmp_path):
    profile = load_telemetry(cic_csv).profile
    path = tmp_path / "profile.json"
    save_profile(profile, path)
    restored = DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored.row_count == profile.row_count
    assert restored.columns[0].name == profile.columns[0].name
