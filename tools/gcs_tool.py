"""tools/gcs_tool.py — GCS upload, download, and checkpoint helpers."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)
_client: storage.Client | None = None


def _get_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client(project=os.environ["PROJECT_ID"])
    return _client


# ── Core helpers ──────────────────────────────────────────────────────────────

def upload_string(bucket_name: str, gcs_path: str, content: str, content_type: str = "text/plain") -> str:
    """Upload a string directly to GCS. Returns gs:// URI."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(content, content_type=content_type)
    uri = f"gs://{bucket_name}/{gcs_path}"
    logger.info(f"Uploaded → {uri}")
    return uri


def upload_file(bucket_name: str, gcs_path: str, local_path: str) -> str:
    """Upload a local file to GCS. Returns gs:// URI."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{gcs_path}"
    logger.info(f"Uploaded {local_path} → {uri}")
    return uri


def download_file(bucket_name: str, gcs_path: str, local_path: str) -> str:
    """Download a GCS file to local path. Returns local path."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{bucket_name}/{gcs_path} → {local_path}")
    return local_path


def download_string(bucket_name: str, gcs_path: str) -> str:
    """Download GCS object as string."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    return blob.download_as_text()


def exists(bucket_name: str, gcs_path: str) -> bool:
    """Check if a GCS object exists."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    return blob.exists()


# ── JSON artifact helpers ──────────────────────────────────────────────────────

def save_json_artifact(bucket_name: str, gcs_path: str, data: dict | list) -> str:
    """Save a dict/list as a JSON artifact to GCS."""
    return upload_string(
        bucket_name,
        gcs_path,
        json.dumps(data, indent=2, ensure_ascii=False),
        content_type="application/json",
    )


def load_json_artifact(bucket_name: str, gcs_path: str) -> dict | list | None:
    """Load a JSON artifact from GCS. Returns None if not found."""
    if not exists(bucket_name, gcs_path):
        return None
    return json.loads(download_string(bucket_name, gcs_path))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def checkpoint_path(prefix: str, phase: str, iteration: int) -> str:
    """Return GCS path for a checkpoint artifact."""
    return f"{prefix}/{phase}/{phase}-iter{iteration}.json"


def load_checkpoint(
    bucket_name: str,
    prefix: str,
    phase: str,
    iteration: int,
    resume: bool = False,
) -> dict | list | None:
    """Load a checkpoint if resume mode is enabled and artifact exists."""
    if not resume:
        return None
    path = checkpoint_path(prefix, phase, iteration)
    data = load_json_artifact(bucket_name, path)
    if data:
        logger.info(f"Resuming from checkpoint: gs://{bucket_name}/{path}")
    return data


def gcs_uri_to_parts(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/path into (bucket, path)."""
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    without_scheme = uri[5:]
    bucket, _, path = without_scheme.partition("/")
    return bucket, path
