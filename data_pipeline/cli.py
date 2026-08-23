"""CLI entry point.

Usage (from repository root):

    python -m data_pipeline.cli --input path/to/flows.csv --out ml_dataset
    python -m data_pipeline.cli --input capture.pcap --out out --profile-only
"""

import argparse
import json
import sys
from pathlib import Path

from .src.pipeline import run_pipeline
from .src.utils.config import PipelineConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="THREATCAST data pipeline")
    parser.add_argument("--input", required=True, help="Telemetry file: .csv / .pcap / .pcapng")
    parser.add_argument("--out", default=None, help="Output directory for ML dataset artifacts")
    parser.add_argument("--window", type=float, default=None, help="TIME_WINDOW_SECONDS override")
    parser.add_argument("--seq-len", type=int, default=None, help="ML_SEQUENCE_LENGTH override")
    parser.add_argument("--horizon", type=int, default=None, help="ML_PREDICTION_HORIZON override")
    parser.add_argument("--source-name", default=None, help="Verified dataset name for label_source")
    parser.add_argument("--profile-only", action="store_true", help="Profile only; skip ML export")
    args = parser.parse_args(argv)

    cfg_kwargs = {
        k: v for k, v in {
            "window_seconds": args.window,
            "sequence_length": args.seq_len,
            "prediction_horizon": args.horizon,
        }.items() if v is not None
    }
    config = PipelineConfig(**cfg_kwargs) if cfg_kwargs else PipelineConfig.from_env()

    try:
        result = run_pipeline(
            args.input,
            None if args.profile_only else args.out,
            config=config,
            source_name=args.source_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "source_type": result.source_type,
        "states": len(result.states),
        "sequences": len(result.sequences),
        "profile_report": result.profile_json_path,
        "artifacts": result.artifacts,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
