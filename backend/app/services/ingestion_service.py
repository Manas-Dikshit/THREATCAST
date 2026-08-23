"""IngestionService: upload validation -> Phase 2 pipeline -> persistence.

Security rules (docs/DEPLOYMENT.md):
- extension whitelist (.csv/.pcap/.pcapng only — what Phase 2 actually parses)
- content sniffing (PCAP magic bytes / CSV text header)
- size cap enforced while streaming to a temp file
- filenames sanitized; files live in a private temp dir, never executed
"""

import csv
import logging
import re
import tempfile
from pathlib import Path

from ..core.config import get_settings
from ..core.errors import InvalidInputError, PayloadTooLargeError, UnsupportedFormatError
from ..models.tables import Dataset, NetworkStateRow
from ..repositories.repositories import (
    DatasetRepository, IngestionJobRepository, StateRepository,
)

logger = logging.getLogger("BACKEND")

ALLOWED_EXTENSIONS = {".csv", ".pcap", ".pcapng"}
PCAP_MAGICS = (
    b"\xa1\xb2\xc3\xd4", b"\xd4\xc3\xb2\xa1",  # pcap LE/BE
    b"\x0a\x0d\x0d\x0a",                        # pcapng
)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class UploadValidator:
    def __init__(self, max_size_mb: int):
        self.max_bytes = max_size_mb * 1024 * 1024

    @staticmethod
    def sanitize_filename(name: str | None) -> str:
        name = Path(name or "upload.bin").name  # kills any path traversal
        cleaned = _SAFE_NAME.sub("_", name).strip("._") or "upload.bin"
        return cleaned[:255]

    @staticmethod
    def detect_extension(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file type '{ext or '(none)'}'. Allowed: CSV, PCAP, PCAPNG.",
                allowed=sorted(ALLOWED_EXTENSIONS),
            )
        return ext

    def sniff(self, ext: str, head: bytes) -> str:
        """Cheap content check: real PCAP magic for binary, printable header for CSV."""
        if ext in (".pcap", ".pcapng"):
            if not any(head.startswith(m) for m in PCAP_MAGICS):
                raise InvalidInputError("File claims to be PCAP but magic bytes do not match.")
            return "pcap"
        if b"\x00" in head:
            raise InvalidInputError("Binary content is not valid CSV.")
        return "csv"

    def save_to_temp(self, upload, workdir: Path) -> tuple[Path, int, str]:
        """Stream the upload into workdir under a sanitized name.
        Returns (path, size_bytes, format)."""
        original = self.sanitize_filename(upload.filename)
        ext = self.detect_extension(original)
        dest = workdir / f"upload_{new_token()}{ext}"
        size = 0
        head = b""
        try:
            with open(dest, "wb") as fh:
                while chunk := upload.file.read(1024 * 1024):
                    if not head:
                        head = chunk[:16]
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise PayloadTooLargeError(
                            f"Upload exceeds {self.max_bytes // (1024*1024)} MB limit."
                        )
                    fh.write(chunk)
        except PayloadTooLargeError:
            dest.unlink(missing_ok=True)
            raise
        if size == 0:
            dest.unlink(missing_ok=True)
            raise InvalidInputError("Uploaded file is empty.")
        fmt = self.sniff(ext, head)
        # validate CSV has at least a parseable header row
        if fmt == "csv":
            try:
                with open(dest, encoding="utf-8-sig", errors="replace") as fh:
                    next(csv.reader(fh))
            except (csv.Error, StopIteration):
                dest.unlink(missing_ok=True)
                raise InvalidInputError("CSV file has no readable header row.") from None
        return dest, size, fmt


def new_token(n: int = 8) -> str:
    import uuid

    return uuid.uuid4().hex[:n]


class IngestionService:
    def __init__(self, db):
        self.db = db
        self.jobs = IngestionJobRepository(db)
        self.datasets = DatasetRepository(db)
        self.states = StateRepository(db)
        self.settings = get_settings()

    def process_upload(self, upload, source_name: str | None,
                       model_service) -> dict:
        """Full synchronous flow. Returns the job-status payload."""
        validator = UploadValidator(self.settings.max_upload_size_mb)
        with tempfile.TemporaryDirectory(prefix="threatcast_upload_") as tmp:
            workdir = Path(tmp)

            path, size, fmt = validator.save_to_temp(upload, workdir)

            job = self.jobs.create(filename=path.name, file_format=fmt)
            self.db.commit()

            self.jobs.set_status(job, "PROCESSING")
            self.db.commit()
            try:
                dataset_id, n_states, n_seqs = self._run_pipeline(
                    path, fmt, size, source_name, workdir, job
                )
                prediction_id = self._predict_latest(job, dataset_id, model_service)
                self.jobs.set_status(job, "COMPLETED")
                self.db.commit()
                return {
                    "job_id": job.id, "status": "completed",
                    "dataset_id": dataset_id,
                    "states_generated": n_states,
                    "sequences_generated": n_seqs,
                    "prediction_id": prediction_id,
                }
            except Exception as exc:
                self.db.rollback()
                job = self.jobs.get(job.id)
                message = _friendly_pipeline_error(exc)
                logger.exception("[%s] ingestion failed", job.id)
                self.jobs.set_status(job, "FAILED", error=message)
                self.db.commit()
                from ..core.errors import JobFailedError

                raise JobFailedError(message) from exc

    def _run_pipeline(self, path: Path, fmt: str, size: int,
                      source_name: str | None, workdir: Path, job) -> tuple[str, int, int]:
        from data_pipeline.src.pipeline import run_pipeline

        export_dir = workdir / "export"
        result = run_pipeline(path, export_dir, source_name=source_name)

        dataset = self.datasets.create(
            filename=path.name, file_format=fmt, size_bytes=size,
            source_name=source_name, storage_path=None,
            row_count=len(result.states),
            meta={"source_type": result.source_type},
        )
        rows = [
            NetworkStateRow(
                state_id=s.state_id, dataset_id=dataset.id,
                timestamp_start=s.timestamp_start, timestamp_end=s.timestamp_end,
                window_seconds=s.window_seconds, features=s.features,
                label=s.label, label_source=s.label_source,
            )
            for s in result.states
        ]
        self.states.add_many(rows)
        job.dataset_id = dataset.id
        job.states_generated = len(rows)
        job.sequences_generated = len(result.sequences)
        self.db.flush()
        return dataset.id, len(rows), len(result.sequences)

    def _predict_latest(self, job, dataset_id: str | None, model_service) -> str | None:
        """Run the world model on the most recent sequence of this dataset."""
        if model_service is None or not model_service.status.loaded:
            logger.info("Model unavailable - skipping post-ingestion prediction")
            return None
        from ..ml_bridge import latest_sequence_for_dataset

        sequence = latest_sequence_for_dataset(self.db, dataset_id)
        if sequence is None:
            return None
        from .prediction_service import PredictionService

        result = PredictionService(self.db, model_service).predict_and_store(
            sequence, dataset_id=dataset_id
        )
        job.meta = {**(job.meta or {}), "prediction_id": result.prediction_id}
        self.db.flush()
        return result.prediction_id


def _friendly_pipeline_error(exc: Exception) -> str:
    text = str(exc)
    if "No usable flow records" in text:
        return "No usable flow records were parsed from the uploaded file."
    if "unknown LL type" in text or "doesn't start with a valid capture file" in text.lower():
        return "File content is not a readable packet capture."
    return "Ingestion failed while processing the uploaded file."


__all__ = ["IngestionService", "UploadValidator"]
