"""Tests for opendrop.inference.llamacpp command construction."""

from __future__ import annotations

from pathlib import Path

from opendrop.inference.llamacpp import LlamaCppServer


def _server(**kwargs):
    return LlamaCppServer(
        gguf_path=Path("/tmp/model.gguf"),
        port=11401,
        binary=Path("/tmp/llama-server"),
        **kwargs,
    )


def test_flash_attn_passes_explicit_on_value():
    # Regression: current llama.cpp builds reject a bare --flash-attn flag
    # ("error while handling argument --flash-attn: expected value for argument").
    cmd = _server(flash_attn=True)._build_cmd()
    assert "--flash-attn" in cmd
    idx = cmd.index("--flash-attn")
    assert cmd[idx + 1] == "on"


def test_flash_attn_omitted_when_disabled():
    cmd = _server(flash_attn=False)._build_cmd()
    assert "--flash-attn" not in cmd


def test_build_cmd_includes_model_port_and_host():
    cmd = _server()._build_cmd()
    assert cmd[cmd.index("--model") + 1] == "/tmp/model.gguf"
    assert cmd[cmd.index("--port") + 1] == "11401"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
