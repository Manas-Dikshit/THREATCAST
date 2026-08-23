"""Ingestion: source detection and unified loading into FlowRecords."""

from .loader import LoadResult, detect_source_type, load_telemetry

__all__ = ["load_telemetry", "detect_source_type", "LoadResult"]
