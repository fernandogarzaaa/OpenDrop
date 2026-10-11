"""Tests for opendrop.inference.llamacpp command construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from opendrop.inference import llamacpp
from opendrop.inference.llamacpp import LlamaCppServer


def _server(**kwargs):
    return LlamaCppServer(
        gguf_path=Path("/tmp/model.gguf"),
        port=11401,
        binary=Path("/tmp/llama-server"),
        **kwargs,
    )


def test_flash_attn_passes_explicit_on_value_for_new_binaries():
    # Current llama.cpp builds reject a bare --flash-attn flag
    # ("error while handling argument --flash-attn: expected value for argument").
    with patch.object(llamacpp, "_flash_attn_takes_value", return_value=True):
        cmd = _server(flash_attn=True)._build_cmd()
    assert "--flash-attn" in cmd
    idx = cmd.index("--flash-attn")
    assert cmd[idx + 1] == "on"


def test_flash_attn_bare_flag_for_pre_9308_binaries():
    # Pre-ggml-org/llama.cpp#9308 builds define --flash-attn as a bare flag
    # and reject a trailing value.
    with patch.object(llamacpp, "_flash_attn_takes_value", return_value=False):
        cmd = _server(flash_attn=True)._build_cmd()
    idx = cmd.index("--flash-attn")
    assert idx == len(cmd) - 1 or not cmd[idx + 1].startswith("on")


def test_flash_attn_omitted_when_disabled():
    cmd = _server(flash_attn=False)._build_cmd()
    assert "--flash-attn" not in cmd


def test_build_cmd_includes_model_port_and_host():
    cmd = _server()._build_cmd()
    assert cmd[cmd.index("--model") + 1] == "/tmp/model.gguf"
    assert cmd[cmd.index("--port") + 1] == "11401"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_probe_detects_value_form_from_real_help():
    # Grounded against the real b11556 binary help text.
    help_text = "-fa,   --flash-attn [on|off|auto]       set Flash Attention use"
    with patch("subprocess.run") as run:
        run.return_value.stdout = help_text
        run.return_value.stderr = ""
        llamacpp._flash_attn_takes_value.cache_clear()
        assert llamacpp._flash_attn_takes_value("/tmp/llama-server") is True


def test_probe_detects_bare_flag_form():
    help_text = "--flash-attn              enable flash attention"
    with patch("subprocess.run") as run:
        run.return_value.stdout = help_text
        run.return_value.stderr = ""
        llamacpp._flash_attn_takes_value.cache_clear()
        assert llamacpp._flash_attn_takes_value("/tmp/llama-server") is False
