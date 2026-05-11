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
