"""Download manager for OpenDrop.

Handles resumable downloads with progress reporting (Rich), SHA-256
verification, and per-file locking to prevent double-downloads.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

_CHUNK = 1024 * 1024  # 1 MB read chunks
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _lock_for(url: str) -> threading.Lock:
    with _LOCKS_LOCK:
        if url not in _LOCKS:
            _LOCKS[url] = threading.Lock()
        return _LOCKS[url]


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = unquote(path.split("/")[-1].split("?")[0])
    return name or "model.bin"


def _file_sha256(path: Path, progress_cb: Callable[[int], None] | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            if progress_cb:
                progress_cb(len(chunk))
    return h.hexdigest()


class DownloadError(RuntimeError):
    pass


def download(
    url: str,
    dest_dir: Path,
    filename: str | None = None,
    expected_sha256: str | None = None,
    token: str | None = None,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download *url* into *dest_dir*, resuming if partially downloaded.

    Args:
        url:             The URL to download.
        dest_dir:        Destination directory.
        filename:        Override the inferred filename.
        expected_sha256: If provided, verify after download.
        token:           HuggingFace auth token.
        force:           Re-download even if file exists and passes hash check.
        show_progress:   Show a Rich progress bar.

    Returns:
        Path to the downloaded file.

    Raises:
        DownloadError: On network failure, HTTP error, or hash mismatch.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or _filename_from_url(url)
    dest = dest_dir / fname
    lock = _lock_for(url)

    with lock:
        # Already complete?
        if dest.exists() and not force:
            if expected_sha256 is None:
                return dest
            if _file_sha256(dest) == expected_sha256:
                return dest

        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Determine resume offset
        resume_at = dest.stat().st_size if dest.exists() else 0
        if resume_at:
            headers["Range"] = f"bytes={resume_at}-"

        progress = Progress(
            TextColumn("[bold blue]{task.fields[name]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            disable=not show_progress,
        )

        try:
            with httpx.stream("GET", url, headers=headers,
                              follow_redirects=True, timeout=60) as resp:
                if resp.status_code not in (200, 206):
                    raise DownloadError(
                        f"HTTP {resp.status_code} downloading {url}"
                    )
                total = int(resp.headers.get("content-length", 0)) + resume_at
                mode = "ab" if resume_at and resp.status_code == 206 else "wb"
                if mode == "wb":
                    resume_at = 0  # server didn't honour Range

                with progress:
                    task: TaskID = progress.add_task(
                        "download",
                        name=fname,
                        total=total or None,
                        completed=resume_at,
                    )
                    with open(dest, mode) as fh:
                        for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                            fh.write(chunk)
                            progress.advance(task, len(chunk))

        except httpx.RequestError as exc:
            raise DownloadError(f"Network error downloading {url}: {exc}") from exc

        if expected_sha256:
            actual = _file_sha256(dest)
            if actual != expected_sha256:
                dest.unlink(missing_ok=True)
                raise DownloadError(
                    f"SHA-256 mismatch for {fname}: "
                    f"expected {expected_sha256}, got {actual}"
                )

        return dest


def download_repo_files(
    model_id: str,
    filenames: list[str],
    dest_dir: Path,
    token: str | None = None,
    show_progress: bool = True,
) -> list[Path]:
    """Download multiple files from a HuggingFace repo.

    Args:
        model_id:  HF model ID, e.g. 'org/model'.
        filenames: List of relative file paths within the repo.
        dest_dir:  Destination directory.
        token:     HuggingFace auth token.

    Returns:
        List of downloaded file paths.
    """
    base = f"https://huggingface.co/{model_id}/resolve/main"
    paths = []
    for fname in filenames:
        url = f"{base}/{fname}"
        paths.append(
            download(url, dest_dir, filename=fname, token=token,
                     show_progress=show_progress)
        )
    return paths
