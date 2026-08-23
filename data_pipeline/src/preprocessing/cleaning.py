"""DataFrame cleaning: duplicates, NaN/inf, invalid values.

Operates on the canonical-mapped flow table. Cleaning never invents values —
it drops or nulls them, and reports what it did.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningStats:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    infinite_values_nulled: int = 0
    malformed_rows_dropped: int = 0
    columns: list[str] = field(default_factory=list)


def clean_flow_table(df: pd.DataFrame, *, drop_duplicate_rows: bool = True) -> tuple[pd.DataFrame, CleaningStats]:
    stats = CleaningStats(rows_in=len(df))
    out = df.copy()

    # 1. Replace +-inf with NaN everywhere numeric.
    numeric = out.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(out[numeric].to_numpy(dtype="float64", na_value=np.nan)) if len(numeric) else []
    stats.infinite_values_nulled = int(inf_mask.sum()) if len(numeric) else 0
    if len(numeric):
        out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)

    # 2. Drop exact duplicate rows.
    before = len(out)
    if drop_duplicate_rows:
        out = out.drop_duplicates()
    stats.duplicates_removed = before - len(out)

    # 3. Drop unusable rows (no parseable timestamp -> cannot be windowed).
    if "timestamp" in out.columns:
        before = len(out)
        out = out.dropna(subset=["timestamp"])
        stats.malformed_rows_dropped = before - len(out)

    # 4. Sort chronologically — downstream windows assume order.
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp").reset_index(drop=True)

    stats.rows_out = len(out)
    stats.columns = list(out.columns)
    return out, stats
