"""Feature extraction: window-level aggregation over normalized flows."""

from .aggregator import aggregate_window, majority_label

__all__ = ["aggregate_window", "majority_label"]
