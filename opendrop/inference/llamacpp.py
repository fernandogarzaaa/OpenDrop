"""llama.cpp subprocess backend for OpenDrop.

Manages one `llama-server` process per model instance.  Provides:
  - Process lifecycle (start / stop / health-check)
  - Port allocation
  - Log capture with Rich
  - Graceful shutdown with SIGTERM → SIGKILL fallback
"""

from __future__ import annotations

import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

_SERVER_NAMES = ["llama-server", "llama_server", "server"]
_QUANTIZE_NAMES = ["llama-quantize", "llama_quantize"]
_COMMON_DIRS = [
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
    Path.home() / "llama.cpp" / "build" / "bin",
    Path.home() / "llama.cpp" / "build",
    Path("/usr/bin"),
]


def find_binary(names: list[str]) -> Path | None:
    """Search PATH and common directories for one of the given binary names."""
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    for d in _COMMON_DIRS:
        for name in names:
            p = d / name
            if p.exists() and p.is_file():
                return p
    return None


def find_server_binary() -> Path | None:
    return find_binary(_SERVER_NAMES)


def find_quantize_binary() -> Path | None:
    return find_binary(_QUANTIZE_NAMES)


def require_server_binary() -> Path:
    b = find_server_binary()
    if not b:
        raise RuntimeError(
            "llama-server binary not found.\n"
            "Install llama.cpp:\n"
            "  macOS (Homebrew): brew install llama.cpp\n"
            "  Manual:           https://github.com/ggml-org/llama.cpp#build\n"
            "Then re-run opendrop."
        )
    return b


# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------

_allocated_ports: set[int] = set()
_PORT_LOCK = threading.Lock()


def _find_free_port(start: int = 11401, end: int = 11500) -> int:
    with _PORT_LOCK:
        for port in range(start, end):
            if port in _allocated_ports:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    _allocated_ports.add(port)
                    return port
                except OSError:
                    continue
    raise RuntimeError("No free port available in range 11401-11500")


def _release_port(port: int) -> None:
    with _PORT_LOCK:
        _allocated_ports.discard(port)


# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

class LlamaCppServer:
    """Manages one llama-server subprocess."""

    def __init__(
        self,
        gguf_path: Path,
        port: int,
        ctx_size: int = 8192,
        gpu_layers: int = -1,
        parallel: int = 1,
        flash_attn: bool = True,
        extra_args: list[str] | None = None,
        binary: Path | None = None,
    ) -> None:
        self.gguf_path = gguf_path
        self.port = port
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.parallel = parallel
        self.flash_attn = flash_attn
        self.extra_args = extra_args or []
        self._binary = binary or require_server_binary()
        self._proc: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None

    def _build_cmd(self) -> list[str]:
        cmd = [
            str(self._binary),
            "--model", str(self.gguf_path),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.gpu_layers),
            "--parallel", str(self.parallel),
        ]
        if self.flash_attn:
            cmd.append("--flash-attn")
        cmd.extend(self.extra_args)
        return cmd

    def start(self, timeout: int = 60) -> None:
        """Start the server process and wait until it is healthy."""
        cmd = self._build_cmd()
        console.print(f"[dim]Starting llama-server on port {self.port}[/dim]")
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs, daemon=True
        )
        self._log_thread.start()
        self._wait_healthy(timeout)

    def _stream_logs(self) -> None:
        if self._proc and self._proc.stdout:
            for line in self._proc.stdout:
                console.print(f"[dim][llama.cpp] {line.rstrip()}[/dim]")

    def _wait_healthy(self, timeout: int) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + timeout
        import httpx
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited unexpectedly (code {self._proc.returncode})"
                )
            try:
                r = httpx.get(url, timeout=2)
                if r.status_code == 200:
                    console.print(
                        f"[green]✓ llama-server ready on port {self.port}[/green]"
                    )
                    return
            except Exception:
                pass
            time.sleep(1)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy within {timeout}s")

    def stop(self) -> None:
        """Gracefully stop the server."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        _release_port(self.port)
        self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return f"<LlamaCppServer port={self.port} status={status} model={self.gguf_path.name}>"


# ---------------------------------------------------------------------------
# Multi-server manager
# ---------------------------------------------------------------------------

class ServerManager:
    """Manages a pool of LlamaCppServer instances keyed by model record ID."""

    def __init__(self) -> None:
        self._servers: dict[str, LlamaCppServer] = {}
        self._lock = threading.Lock()

    def start_model(
        self,
        model_record_id: str,
        gguf_path: Path,
        ctx_size: int = 8192,
        gpu_layers: int = -1,
        parallel: int = 1,
        flash_attn: bool = True,
        port: int | None = None,
    ) -> LlamaCppServer:
        with self._lock:
            if model_record_id in self._servers:
                srv = self._servers[model_record_id]
                if srv.is_running():
                    return srv
                del self._servers[model_record_id]

            p = port or _find_free_port()
            srv = LlamaCppServer(
                gguf_path=gguf_path,
                port=p,
                ctx_size=ctx_size,
                gpu_layers=gpu_layers,
                parallel=parallel,
                flash_attn=flash_attn,
            )
            srv.start()
            self._servers[model_record_id] = srv
            return srv

    def stop_model(self, model_record_id: str) -> None:
        with self._lock:
            srv = self._servers.pop(model_record_id, None)
            if srv:
                srv.stop()

    def stop_all(self) -> None:
        with self._lock:
            for srv in self._servers.values():
                srv.stop()
            self._servers.clear()

    def get_server(self, model_record_id: str) -> LlamaCppServer | None:
        return self._servers.get(model_record_id)

    def running_models(self) -> dict[str, LlamaCppServer]:
        with self._lock:
            return {k: v for k, v in self._servers.items() if v.is_running()}


# Module-level singleton
_manager = ServerManager()


def get_manager() -> ServerManager:
    return _manager
