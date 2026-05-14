"""Tests for opendrop.inference.server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from opendrop.inference.server import create_app


@pytest.fixture
def mock_registry():
    reg = AsyncMock()
    reg.init = AsyncMock()
    reg.list_models = AsyncMock(return_value=[])
    reg.get_model = AsyncMock(return_value=None)
    reg.touch_model = AsyncMock()
    return reg


@pytest.fixture
def mock_manager():
    mgr = MagicMock()
    mgr.running_models.return_value = {}
    mgr.stop_all = MagicMock()
    return mgr


@pytest.fixture
def client(mock_registry, mock_manager):
    app = create_app(manager=mock_manager, registry=mock_registry)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestModelsEndpoint:
    def test_list_models_empty(self, client: TestClient, mock_registry):
        mock_registry.list_models.return_value = []
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert data["data"] == []


class TestChatCompletions:
    def test_model_not_found_returns_404(self, client: TestClient, mock_registry):
        mock_registry.get_model.return_value = None
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert r.status_code == 404

    def test_missing_model_file_returns_503(self, client: TestClient, mock_registry):
        from datetime import datetime, timezone

        from opendrop.core.registry import ModelRecord

        rec = ModelRecord(
            id="test-model",
            model_id="org/model",
            source_url="",
            display_name="test-model",
            architecture="llama",
            params_b=7.0,
            quant="Q4_K_M",
            format="gguf",
            path="/nonexistent/model.gguf",
            size_bytes=0,
            license_id="apache-2.0",
            license_warning="",
            tags=[],
            pipeline_tag="",
            added_at=datetime.now(timezone.utc).isoformat(),
            last_used=None,
            server_port=None,
            extra={},
        )
        mock_registry.get_model.return_value = rec

        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert r.status_code == 503


class TestWebUI:
    def test_web_ui_served(self, client: TestClient):

        from opendrop.inference.server import create_app
        from opendrop.ui.web import mount_web_ui

        app = create_app()
        mount_web_ui(app)
        with TestClient(app) as c:
            r = c.get("/")
            assert r.status_code == 200
            assert "OpenDrop" in r.text


class TestHardwareEndpoint:
    def test_hardware_returns_200(self, client: TestClient, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import opendrop.inference.server as server_mod

        fake_gpu = SimpleNamespace(kind=MagicMock(value="none"), name="", unified=False)
        fake_hw = SimpleNamespace(
            os_name="Linux",
            cpu_arch="x86_64",
            cpu_cores=8,
            cpu_physical_cores=4,
            ram_mb=16384,
            free_ram_mb=8192,
            effective_memory_mb=4915,
            gpu=fake_gpu,
            usable_vram_mb=0,
            backend_priority=["cpu"],
            has_avx2=True,
            has_neon=False,
            ssd_est_mb_per_s=500,
        )
        monkeypatch.setattr(server_mod, "detect_hardware", lambda: fake_hw)
        r = client.get("/v1/hardware")
        assert r.status_code == 200
        data = r.json()
        assert data["os"] == "Linux"
        assert data["cpu_arch"] == "x86_64"
        assert data["cpu_cores"] == 8
        assert data["ram_mb"] == 16384
        assert data["has_avx2"] is True
        assert data["has_neon"] is False
        assert "effective_memory_mb" in data
        assert "backend_priority" in data

    def test_hardware_keys_present(self, client: TestClient, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import opendrop.inference.server as server_mod

        fake_gpu = SimpleNamespace(
            kind=MagicMock(value="apple_silicon"), name="M3 Max", unified=True
        )
        fake_hw = SimpleNamespace(
            os_name="Darwin",
            cpu_arch="arm64",
            cpu_cores=12,
            cpu_physical_cores=12,
            ram_mb=32768,
            free_ram_mb=20000,
            effective_memory_mb=24576,
            gpu=fake_gpu,
            usable_vram_mb=32768,
            backend_priority=["metal", "cpu"],
            has_avx2=False,
            has_neon=True,
            ssd_est_mb_per_s=2000,
        )
        monkeypatch.setattr(server_mod, "detect_hardware", lambda: fake_hw)
        r = client.get("/v1/hardware")
        assert r.status_code == 200
        data = r.json()
        expected_keys = {
            "os",
            "cpu_arch",
            "cpu_cores",
            "cpu_physical_cores",
            "ram_mb",
            "free_ram_mb",
            "effective_memory_mb",
            "gpu_kind",
            "gpu_name",
            "gpu_unified",
            "usable_vram_mb",
            "backend_priority",
            "has_avx2",
            "has_neon",
            "ssd_est_mb_per_s",
        }
        assert expected_keys.issubset(data.keys())
        assert data["gpu_name"] == "M3 Max"
        assert data["gpu_unified"] is True


class TestPullEndpoint:
    def test_pull_missing_source_returns_422(self, client: TestClient):
        r = client.post("/v1/pull", json={})
        assert r.status_code == 422

    def test_pull_streams_sse(self, client: TestClient, monkeypatch):
        import opendrop.inference.server as server_mod

        class _FakeOrch:
            def pull(self, source, token=None, quant_override=None, **_kw):
                pass  # no-op, sentinel will fire via finally

        monkeypatch.setattr(server_mod, "Orchestrator", _FakeOrch)
        r = client.post(
            "/v1/pull",
            json={"source": "org/model"},
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
