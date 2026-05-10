"""Tests for opendrop.core.downloader."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from opendrop.core.downloader import (
    DownloadError,
    _filename_from_url,
    _file_sha256,
    download,
)


class TestFilenameFromUrl:
    def test_simple_gguf(self):
        assert _filename_from_url(
            "https://huggingface.co/org/model/resolve/main/model.Q4_K_M.gguf"
        ) == "model.Q4_K_M.gguf"

    def test_with_query_string(self):
        name = _filename_from_url("https://example.com/file.bin?token=abc")
        assert name == "file.bin"

    def test_fallback(self):
        name = _filename_from_url("https://example.com/")
        assert name == "model.bin"


class TestFileSha256:
    def test_sha256_correct(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        content = b"hello world" * 1000
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _file_sha256(f) == expected


class TestDownload:
    def _make_mock_response(self, content: bytes, status: int = 200):
        """Build a mock httpx streaming response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.headers = {"content-length": str(len(content))}
        mock_resp.iter_bytes.return_value = [content]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("opendrop.core.downloader.httpx.stream")
    def test_download_success(self, mock_stream, tmp_path: Path):
        content = b"model data " * 100
        mock_resp = self._make_mock_response(content)
        mock_stream.return_value = mock_resp

        result = download(
            "https://example.com/model.gguf",
            tmp_path,
            show_progress=False,
        )
        assert result.name == "model.gguf"
        assert result.read_bytes() == content

    @patch("opendrop.core.downloader.httpx.stream")
    def test_download_http_error_raises(self, mock_stream, tmp_path: Path):
        mock_resp = self._make_mock_response(b"error", status=404)
        mock_stream.return_value = mock_resp

        with pytest.raises(DownloadError):
            download("https://example.com/notfound.gguf", tmp_path, show_progress=False)

    @patch("opendrop.core.downloader.httpx.stream")
    def test_download_sha256_mismatch_raises(self, mock_stream, tmp_path: Path):
        content = b"some data"
        mock_resp = self._make_mock_response(content)
        mock_stream.return_value = mock_resp

        with pytest.raises(DownloadError, match="SHA-256"):
            download(
                "https://example.com/model.gguf",
                tmp_path,
                expected_sha256="deadbeef" * 8,
                show_progress=False,
            )

    @patch("opendrop.core.downloader.httpx.stream")
    def test_download_sha256_match_succeeds(self, mock_stream, tmp_path: Path):
        content = b"valid data bytes"
        expected = hashlib.sha256(content).hexdigest()
        mock_resp = self._make_mock_response(content)
        mock_stream.return_value = mock_resp

        result = download(
            "https://example.com/model.gguf",
            tmp_path,
            expected_sha256=expected,
            show_progress=False,
        )
        assert result.exists()

    def test_download_skips_existing_without_hash(self, tmp_path: Path):
        existing = tmp_path / "model.gguf"
        existing.write_bytes(b"existing data")

        with patch("opendrop.core.downloader.httpx.stream") as mock_stream:
            result = download(
                "https://example.com/model.gguf",
                tmp_path,
                show_progress=False,
            )
            mock_stream.assert_not_called()
        assert result == existing
