"""Tests for opendrop.config validation and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from opendrop.config import load_config


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestLoadConfigValidation:
    def test_load_defaults_when_file_missing(self, tmp_path: Path):
        cfg = load_config(tmp_path / "missing.toml")
        assert cfg.server.port == 11400
        assert cfg.inference.context_size == 8192

    def test_invalid_server_port_raises(self, tmp_path: Path):
        cfg_path = tmp_path / "config.toml"
        _write_toml(
            cfg_path,
            """
[server]
port = 70000
""",
        )
        with pytest.raises(ValueError, match=r"\[server\]\.port"):
            load_config(cfg_path)

    def test_invalid_training_method_raises(self, tmp_path: Path):
        cfg_path = tmp_path / "config.toml"
        _write_toml(
            cfg_path,
            """
[training]
default_method = "invalid-method"
""",
        )
        with pytest.raises(ValueError, match=r"\[training\]\.default_method"):
            load_config(cfg_path)

    def test_invalid_inference_parallel_raises(self, tmp_path: Path):
        cfg_path = tmp_path / "config.toml"
        _write_toml(
            cfg_path,
            """
[inference]
parallel = 0
""",
        )
        with pytest.raises(ValueError, match=r"\[inference\]\.parallel"):
            load_config(cfg_path)
