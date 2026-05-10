"""Tests for opendrop.core.resolver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opendrop.core.resolver import (
    _check_license,
    _extract_hf_model_id,
    _is_direct_file_url,
    _is_hf_url,
    _params_from_name,
    _parse_quant_from_filename,
    resolve,
    search_models,
)


class TestHelpers:
    def test_parse_quant_q4_km(self):
        assert _parse_quant_from_filename("model.Q4_K_M.gguf") == "Q4_K_M"

    def test_parse_quant_iq2(self):
        assert _parse_quant_from_filename("model.IQ2_XXS.gguf") == "IQ2_XXS"

    def test_parse_quant_fp16(self):
        assert _parse_quant_from_filename("model.fp16.gguf") == "FP16"

    def test_parse_quant_none(self):
        assert _parse_quant_from_filename("model.bin") == ""

    def test_params_from_name_8b(self):
        assert _params_from_name("Meta-Llama-3-8B-Instruct") == 8.0

    def test_params_from_name_70b(self):
        assert _params_from_name("Llama-3-70B") == 70.0

    def test_params_from_name_unknown(self):
        assert _params_from_name("no-number-here") == 0.0

    def test_check_license_open(self):
        ok, warn = _check_license("apache-2.0")
        assert ok is True
        assert warn == ""

    def test_check_license_unknown(self):
        ok, warn = _check_license("custom-strange-license")
        assert ok is True
        assert warn  # has a warning

    def test_check_license_copyleft(self):
        ok, warn = _check_license("gpl-3.0")
        assert ok is True
        assert "copyleft" in warn.lower() or "gpl" in warn.lower()

    def test_is_hf_url(self):
        assert _is_hf_url("https://huggingface.co/org/model") is True
        assert _is_hf_url("https://example.com/model") is False

    def test_extract_hf_model_id(self):
        assert _extract_hf_model_id("https://huggingface.co/org/model") == "org/model"
        assert (
            _extract_hf_model_id(
                "https://huggingface.co/org/model/resolve/main/f.gguf"
            ) == "org/model"
        )

    def test_is_direct_file_url(self):
        assert _is_direct_file_url("https://hf.co/x/y/resolve/main/m.gguf") is True
        assert _is_direct_file_url("https://hf.co/x/y") is False


class TestResolveLocal:
    def test_resolve_local_gguf(self, tmp_path: Path):
        gguf = tmp_path / "model.Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 100)
        spec = resolve(str(gguf))
        assert spec.is_local
        assert spec.direct_file is not None
        assert spec.direct_file.is_gguf
        assert spec.direct_file.quant_label == "Q4_K_M"

    def test_resolve_local_nonexistent_raises(self):
        with pytest.raises((ValueError, Exception)):
            resolve("/nonexistent/path/model.gguf")


class TestResolveHF:
    def _mock_hf(self):
        """Return a mock for _hf_model_info and _hf_model_files."""
        info = {
            "config": {"model_type": "llama"},
            "tags": ["license:apache-2.0", "text-generation"],
            "pipeline_tag": "text-generation",
            "cardData": {"license": "apache-2.0"},
        }
        tree = [
            {"type": "file", "path": "model.Q4_K_M.gguf", "size": 4_500_000_000},
            {"type": "file", "path": "model.Q8_0.gguf",   "size": 9_000_000_000},
            {"type": "file", "path": "config.json",        "size": 1024},
        ]
        return info, tree

    @patch("opendrop.core.resolver._hf_model_files")
    @patch("opendrop.core.resolver._hf_model_info")
    def test_resolve_hf_url(self, mock_info, mock_files):
        info, tree = self._mock_hf()
        mock_info.return_value = info
        mock_files.return_value = tree

        spec = resolve("https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        assert spec.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert spec.architecture == "llama"
        assert spec.license_id == "apache-2.0"
        assert spec.license_ok is True
        assert spec.license_warning == ""
        assert len([v for v in spec.variants if v.is_gguf]) == 2

    @patch("opendrop.core.resolver._hf_model_files")
    @patch("opendrop.core.resolver._hf_model_info")
    def test_resolve_bare_model_id(self, mock_info, mock_files):
        info, tree = self._mock_hf()
        mock_info.return_value = info
        mock_files.return_value = tree

        spec = resolve("meta-llama/Meta-Llama-3-8B-Instruct")
        assert spec.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"

    @patch("opendrop.core.resolver._hf_model_files")
    @patch("opendrop.core.resolver._hf_model_info")
    def test_resolve_params_inferred(self, mock_info, mock_files):
        info, tree = self._mock_hf()
        mock_info.return_value = info
        mock_files.return_value = tree

        spec = resolve("bartowski/Meta-Llama-3-8B-Instruct-GGUF")
        assert spec.params_b == 8.0

    @patch("opendrop.core.resolver._hf_model_files")
    @patch("opendrop.core.resolver._hf_model_info")
    def test_best_gguf_sorted_by_size(self, mock_info, mock_files):
        info, tree = self._mock_hf()
        mock_info.return_value = info
        mock_files.return_value = tree

        spec = resolve("org/model")
        best = spec.best_gguf_variants()
        assert best[0].size_bytes >= best[-1].size_bytes

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            resolve("not-a-url-or-path")


class TestSearchModels:
    @patch("opendrop.core.resolver.httpx.Client")
    def test_search_models_parses_hits(self, mock_client):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "downloads": 1234,
                "likes": 77,
                "pipeline_tag": "text-generation",
                "tags": ["license:llama3.1"],
                "lastModified": "2026-01-01T00:00:00.000Z",
            }
        ]
        mock_http = MagicMock()
        mock_http.get.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_http

        results = search_models("llama", limit=5)

        assert len(results) == 1
        assert results[0].model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert results[0].downloads == 1234
        assert results[0].likes == 77
        assert results[0].pipeline_tag == "text-generation"
        assert results[0].license_id == "llama3.1"
