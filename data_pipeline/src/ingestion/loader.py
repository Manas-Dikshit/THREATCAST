"""Source detection and unified loading: file -> FlowRecords + DatasetProfile."""

from dataclasses import dataclass
from pathlib import Path

from ..parsers.csv_flows import build_flow_table, read_raw_csv, records_from_table
from ..parsers.pcap_parser import parse_pcap
from ..preprocessing.cleaning import clean_flow_table
from ..preprocessing.profiler import profile_dataframe
from ..schemas.metadata import DatasetProfile, LabelKind
from ..schemas.records import FlowRecord


@dataclass
class LoadResult:
    records: list[FlowRecord]
    profile: DatasetProfile
    source_type: str  # 'csv' | 'cic_ids2018_csv' | 'pcap'


def detect_source_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in (".pcap", ".pcapng", ".cap"):
        return "pcap"
    if suffix == ".csv":
        return "cic_ids2018_csv" if _looks_like_cic(path) else "csv"
    raise ValueError(f"Unsupported input type '{suffix}'. Supported: .csv, .pcap, .pcapng")


_CIC_SIGNATURES = {"flowduration", "flowiatmean", "flowbyts", "flowpkts", "totlenfwdpkts", "bwdpktlens"}


def _looks_like_cic(path: Path) -> bool:
    """Sniff the first line for CIC-style headers (never trusted blindly — just a hint)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    from ..parsers.columns import normalize_header

    tokens = {normalize_header(token) for token in first.split(",")}
    return len(tokens & _CIC_SIGNATURES) >= 2


def load_telemetry(path: str | Path) -> LoadResult:
    """Load any supported telemetry source into normalized FlowRecords."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    source_type = detect_source_type(path)

    if source_type == "pcap":
        records, profile = parse_pcap(path)
        return LoadResult(records=records, profile=profile, source_type="pcap")

    raw = read_raw_csv(path)
    profile = profile_dataframe(raw, source_file=str(path), detected_format=source_type)
    table, mapping, warnings = build_flow_table(raw)
    profile.mapped_columns = mapping
    profile.unmapped_columns = [c for c in raw.columns if c not in mapping.values()]
    profile.warnings.extend(warnings)

    cleaned, stats = clean_flow_table(table)
    if stats.duplicates_removed:
        profile.warnings.append(f"Removed {stats.duplicates_removed} duplicate rows during normalization.")
    if stats.infinite_values_nulled:
        profile.warnings.append(f"Nulled {stats.infinite_values_nulled} infinite values.")
    if stats.malformed_rows_dropped:
        profile.warnings.append(f"Dropped {stats.malformed_rows_dropped} rows without parseable timestamps.")

    if "timestamp" in cleaned.columns:
        valid = cleaned["timestamp"].dropna()
        if len(valid):
            profile.timestamp_min = valid.min().to_pydatetime()
            profile.timestamp_max = valid.max().to_pydatetime()

    label_kind = LabelKind.GROUND_TRUTH if "label" in cleaned.columns else LabelKind.UNKNOWN
    records = records_from_table(cleaned, label_kind=label_kind)

    # Keep the raw-file label distribution in the profile (ground truth provenance).
    if "label" in table.columns:
        profile.label_distribution = (
            table["label"].value_counts(dropna=True).astype(int).to_dict()
        )

    return LoadResult(records=records, profile=profile, source_type=source_type)


__all__ = ["load_telemetry", "detect_source_type", "LoadResult"]
