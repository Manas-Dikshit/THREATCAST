"""Upload validation: type, size, content, traversal, empty."""

import io

from tests.conftest import make_csv


def _post(client, data: bytes, filename: str, **extra):
    return client.post(
        "/api/v1/ingestion/upload",
        files={"file": (filename, io.BytesIO(data))},
        data={"source_type": extra.get("source_type", "")} if "source_type" in extra else None,
    )


def test_unsupported_extension_rejected(client):
    r = _post(client, b"not an exe really", "malware.exe")
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "UNSUPPORTED_FORMAT"


def test_pcap_extension_with_garbage_content_rejected(client):
    r = _post(client, b"\x00\x01\x02 not a pcap", "fake.pcap")
    assert r.status_code == 422
    assert r.json()["error"]["code"] in ("INVALID_INPUT",)


def test_csv_with_binary_content_rejected(client):
    r = _post(client, b"header,\x00\x01binary", "data.csv")
    assert r.status_code == 422


def test_empty_file_rejected(client):
    r = _post(client, b"", "empty.csv")
    assert r.status_code == 422


def test_path_traversal_sanitized(client):
    """Filename is sanitized; traversal attempt must not escape the temp dir."""
    r = _post(client, make_csv(), "../../evil.csv")
    # Either rejected by extension logic or processed safely under a clean name;
    # it must never be a server error and never leak a path.
    assert r.status_code in (202, 422)
    if r.status_code == 202:
        job = client.get(f"/api/v1/ingestion/jobs/{r.json()['job_id']}").json()
        assert ".." not in job["filename"]
        assert "/" not in job["filename"] and "\\" not in job["filename"]


def test_oversize_rejected(client, monkeypatch):
    from app.services.ingestion_service import UploadValidator
    from app.core.errors import PayloadTooLargeError
    import pytest

    v = UploadValidator(max_size_mb=1)
    v.max_bytes = 10  # shrink limit so our small payload trips it

    class FakeUpload:
        filename = "big.csv"

        def read(self, n=-1):
            return b"x" * 100

    with pytest.raises(PayloadTooLargeError):
        v.save_to_temp(FakeUpload(), __import__("pathlib").Path(__import__("tempfile").mkdtemp()))


def test_upload_success_returns_job(client, csv_bytes):
    r = _post(client, csv_bytes, "flows.csv")
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"].startswith("job_")
    assert body["status"] == "queued"
