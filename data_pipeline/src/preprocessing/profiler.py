"""Dataset profiling: dynamic inspection of actual data.

Produces a DatasetProfile report describing exactly what was found — dtypes,
nulls, infinities, duplicates, label distribution, timestamp range — plus the
canonical column mapping. Used for CIC-IDS2018 inspection and any other input.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from ..schemas.metadata import ColumnProfile, DatasetProfile


def profile_dataframe(
    df: pd.DataFrame,
    *,
    source_file: str | None = None,
    detected_format: str = "csv",
    mapping: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    max_samples: int = 3,
) -> DatasetProfile:
    profile = DatasetProfile(
        source_file=source_file,
        detected_format=detected_format,
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        mapped_columns=mapping or {},
        warnings=warnings or [],
    )

    for column in df.columns:
        series = df[column]
        col = ColumnProfile(
            name=str(column),
            dtype=str(series.dtype),
            null_count=int(series.isna().sum()),
            unique_count=int(series.nunique(dropna=True)),
        )
        col.null_fraction = round(col.null_count / len(df), 6) if len(df) else 0.0

        numeric = pd.to_numeric(series, errors="coerce") if series.dtype == object else series
        if pd.api.types.is_numeric_dtype(numeric):
            values = numeric.dropna().to_numpy(dtype="float64")
            finite = values[np.isfinite(values)] if len(values) else values
            col.inf_count = int(len(values) - len(finite))
            if len(finite):
                col.min_value = float(finite.min())
                col.max_value = float(finite.max())
                col.mean_value = float(finite.mean())
                col.std_value = float(finite.std())

        samples = series.dropna().unique()[:max_samples]
        col.sample_values = [str(s)[:60] for s in samples]
        profile.columns.append(col)

    ts_col = next((c for c in ("timestamp", "Timestamp", "ts") if c in df.columns), None)
    if ts_col is not None:
        parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True, dayfirst=True).dropna()
        if len(parsed):
            profile.timestamp_min = parsed.min().to_pydatetime()
            profile.timestamp_max = parsed.max().to_pydatetime()

    label_col = next((c for c in ("label", "Label", "class") if c in df.columns), None)
    if label_col is not None:
        profile.label_distribution = (
            df[label_col].value_counts(dropna=True).astype(int).to_dict()
        )
    return profile


def save_profile(profile: DatasetProfile, path) -> str:
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


__all__ = ["profile_dataframe", "save_profile"]
