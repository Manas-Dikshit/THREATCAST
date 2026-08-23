"""CSV flow-record parsing: generic flow CSVs, NetFlow/IPFIX-style CSV exports
and CIC-IDS2018 files share one dynamic-mapping code path.

The parser never assumes a fixed schema: columns are discovered via alias
mapping, values are coerced defensively, malformed rows are dropped and
reported, and unavailable fields stay None.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from ..schemas.metadata import DatasetProfile, LabelKind
from ..schemas.records import FlowRecord
from .columns import (
    MICROSECOND_COLUMNS,
    map_columns,
    normalize_header,
    normalize_protocol,
)

_TIMESTAMP_FORMATS = [
    "%d/%m/%Y %H:%M:%S",        # CIC-IDS2018 style
    "%d/%m/%Y %H:%M:%S %p",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]

_FLAG_LETTERS = {
    "flag_fin": "F",
    "flag_syn": "S",
    "flag_rst": "R",
    "flag_psh": "P",
    "flag_ack": "A",
    "flag_urg": "U",
}


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Cumulatively apply explicit formats (day-first first), then inference.

    Each format fills only the rows it can parse, so mixed-format files and
    ambiguous ties don't silently drop rows.
    """
    combined = pd.to_datetime(pd.Series(pd.NaT, index=series.index), errors="coerce", utc=True)
    for fmt in _TIMESTAMP_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce", utc=True)
        combined = combined.fillna(parsed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        inferred = pd.to_datetime(series, errors="coerce", utc=True)
    return combined.fillna(inferred)


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV tolerating BOMs / whitespace; headers kept verbatim."""
    return pd.read_csv(path, low_memory=False, skipinitialspace=True)


def build_flow_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Map raw columns onto canonical names.

    Returns (flow_table, mapping, warnings). The table contains only canonical
    columns that were actually found — missing ones simply do not exist here.
    """
    mapping = map_columns(raw.columns)
    warnings: list[str] = []
    if not mapping:
        raise ValueError(
            "No recognizable flow columns found. Headers: "
            f"{list(raw.columns)[:20]}"
        )
    if "timestamp" not in mapping:
        warnings.append("No timestamp column recognized; windowing impossible.")

    df = raw.rename(columns={orig: canon for canon, orig in mapping.items()}).copy()

    # Coerce every non-timestamp/non-IP/non-label column to numeric where possible.
    ip_label_cols = {"src_ip", "dst_ip", "label", "flags"}
    for col in df.columns:
        if col in ("timestamp", *ip_label_cols):
            continue
        if col in ("protocol",) or col.startswith("flag_"):
            if col.startswith("flag_"):
                _to_numeric(df, [col])
            elif col == "protocol":
                df[col] = df[col].map(normalize_protocol)
            continue
        _to_numeric(df, [col])

    if "timestamp" in df.columns:
        df["timestamp"] = _parse_timestamps(df["timestamp"])

    # Unit normalization: CIC duration/IAT columns are microseconds.
    us_cols = [c for c in df.columns if normalize_header(mapping.get(c, c)) in MICROSECOND_COLUMNS]
    for col in us_cols:
        df[col] = df[col] / 1_000_000.0

    # Flags string from per-flag count columns when present.
    flag_cols_present = [c for c in _FLAG_LETTERS if c in df.columns]
    if flag_cols_present:

        def _flags(row) -> str | None:
            letters = "".join(
                _FLAG_LETTERS[c] for c in flag_cols_present
                if pd.notna(row[c]) and row[c] > 0
            )
            return letters or None

        df["flags"] = df[flag_cols_present].apply(_flags, axis=1)
        df = df.drop(columns=flag_cols_present)

    # total_* fallback: sum forward/backward components when totals absent.
    # Non-finite components count as missing (sum over observed values only).
    def _finite(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if "total_bytes" not in df.columns and {"fwd_bytes", "bwd_bytes"} <= set(df.columns):
        df["total_bytes"] = _finite(df["fwd_bytes"]).fillna(0) + _finite(df["bwd_bytes"]).fillna(0)
    if "total_packets" not in df.columns and {"fwd_packets", "bwd_packets"} <= set(df.columns):
        df["total_packets"] = _finite(df["fwd_packets"]).fillna(0) + _finite(df["bwd_packets"]).fillna(0)
    if "iat_var" not in df.columns and "iat_std" in df.columns:
        df["iat_var"] = df["iat_std"] ** 2

    unmapped = [c for c in raw.columns if c not in mapping.values()]
    return df, mapping, warnings


def records_from_table(df: pd.DataFrame, *, label_kind: LabelKind = LabelKind.UNKNOWN) -> list[FlowRecord]:
    """Convert a cleaned canonical table into FlowRecords. Malformed rows are skipped."""
    records: list[FlowRecord] = []
    for row in df.itertuples(index=False):
        data = row._asdict()
        ts = data.get("timestamp")
        if ts is None or pd.isna(ts):
            continue  # no parseable timestamp -> unusable for temporal windows
        record = FlowRecord(
            timestamp=ts.to_pydatetime(),
            src_ip=_opt_str(data.get("src_ip")),
            dst_ip=_opt_str(data.get("dst_ip")),
            src_port=_opt_int(data.get("src_port")),
            dst_port=_opt_int(data.get("dst_port")),
            protocol=data.get("protocol"),
            total_bytes=_opt_float(data.get("total_bytes")),
            total_packets=_opt_float(data.get("total_packets")),
            duration_s=_opt_float(data.get("duration")),
            flags=data.get("flags") if isinstance(data.get("flags"), str) else None,
            iat_mean_s=_opt_float(data.get("iat_mean")),
            iat_var_s=_opt_float(data.get("iat_var")),
            iat_max_s=_opt_float(data.get("iat_max")),
            fwd_bytes=_opt_float(data.get("fwd_bytes")),
            bwd_bytes=_opt_float(data.get("bwd_bytes")),
            fwd_packets=_opt_float(data.get("fwd_packets")),
            bwd_packets=_opt_float(data.get("bwd_packets")),
            label=_opt_str(data.get("label")),
            label_kind=label_kind if _opt_str(data.get("label")) else LabelKind.UNKNOWN,
        )
        records.append(record)
    return records


def _opt_str(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    text = str(v).strip()
    return text or None


def _opt_int(v) -> int | None:
    try:
        if v is None or pd.isna(v):
            return None
    except TypeError:
        pass
    try:
        port = int(float(str(v).strip()))
    except (ValueError, TypeError):
        return None
    return port if 0 <= port <= 65535 else None


def _opt_float(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
    except TypeError:
        pass
    try:
        value = float(str(v).strip())
    except (ValueError, TypeError):
        return None
    return None if np.isinf(value) else value


def parse_flow_csv(path: str | Path) -> tuple[list[FlowRecord], DatasetProfile]:
    """Full CSV path: raw read -> dynamic mapping -> FlowRecords + profile."""
    path = Path(path)
    raw = read_raw_csv(path)
    profile = profile_from_raw(raw, source_file=str(path), mapping=None, warnings=[])
    df, mapping, warnings = build_flow_table(raw)
    profile.mapped_columns = mapping
    profile.unmapped_columns = [c for c in raw.columns if c not in mapping.values()]
    profile.warnings.extend(warnings)
    if "timestamp" in df.columns:
        valid_ts = df["timestamp"].dropna()
        if len(valid_ts):
            profile.timestamp_min = valid_ts.min().to_pydatetime()
            profile.timestamp_max = valid_ts.max().to_pydatetime()
        dropped = int(df["timestamp"].isna().sum())
        if dropped:
            profile.warnings.append(f"{dropped} rows had unparseable timestamps.")
    if "label" in df.columns:
        profile.label_distribution = df["label"].value_counts(dropna=True).astype(int).to_dict()
    records = records_from_table(df, label_kind=LabelKind.GROUND_TRUTH if "label" in df.columns else LabelKind.UNKNOWN)
    return records, profile


def profile_from_raw(raw: pd.DataFrame, *, source_file, mapping, warnings) -> DatasetProfile:
    """Profile the RAW file before any cleaning (the point of profiling)."""
    from ..preprocessing.profiler import profile_dataframe

    profile = profile_dataframe(raw, source_file=source_file, detected_format="csv")
    if mapping:
        profile.mapped_columns = mapping
    if warnings:
        profile.warnings.extend(warnings)
    return profile
