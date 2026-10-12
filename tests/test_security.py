"""Security tests for the local API server and fine-tuning (audit, Oct 2026)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from opendrop.inference import security as sec
from opendrop.inference.server import create_app
from opendrop.training import finetune as ft

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost",
        "http://localhost:11400",
        "https://localhost:8443",
        "http://127.0.0.1:11400",
        "http://127.10.0.3:9",
        "http://[::1]:11400",
        "  http://127.0.0.1:11400  ",
    ],
)
def test_loopback_origins_allowed(origin: str) -> None:
    assert sec.is_allowed_origin(origin) is True


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example",
        "http://evil.example/http://localhost",
        "http://localhost:11400/",  # path suffix is not a bare origin
        "ftp://localhost",
        "null",
        "",
        "   ",
        "http://192.168.1.10:11400",
    ],
)
def test_foreign_origins_refused(origin: str) -> None:
    assert sec.is_allowed_origin(origin) is False


def test_extra_origins_allowed_exactly() -> None:
    extra = frozenset({"https://app.example"})
    assert sec.is_allowed_origin("https://app.example", extra) is True
    assert sec.is_allowed_origin("https://app.example/", extra) is True
    assert sec.is_allowed_origin("https://other.example", extra) is False
    # "null" (sandboxed iframes, file://) is refused even if someone lists it.
    assert sec.is_allowed_origin("null", frozenset({"null"})) is False


def test_extra_origins_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sec.CORS_ORIGINS_ENV, " https://a.example/ , ,https://b.example")
    assert sec.extra_origins_from_env() == frozenset({"https://a.example", "https://b.example"})
    monkeypatch.delenv(sec.CORS_ORIGINS_ENV)
    assert sec.extra_origins_from_env() == frozenset()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sec.API_KEY_ENV, raising=False)
    assert sec.api_key_from_env() is None
    monkeypatch.setenv(sec.API_KEY_ENV, "   ")
    assert sec.api_key_from_env() is None
    monkeypatch.setenv(sec.API_KEY_ENV, " s3cret ")
    assert sec.api_key_from_env() == "s3cret"


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1", "[::1]"]
)
def test_loopback_hosts(host: str) -> None:
    assert sec.is_loopback_host(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com", ""])
def test_non_loopback_hosts(host: str) -> None:
    assert sec.is_loopback_host(host) is False


def test_presented_key_sources() -> None:
    assert sec.presented_key("Bearer abc", None) == "abc"
    assert sec.presented_key("bearer  abc ", None) == "abc"
    assert sec.presented_key("Basic abc", None) is None
    assert sec.presented_key("Bearer ", None) is None
    assert sec.presented_key(None, " xyz ") == "xyz"
    assert sec.presented_key("Basic abc", "xyz") == "xyz"
    assert sec.presented_key(None, "  ") is None
    assert sec.presented_key(None, None) is None


def test_key_matches() -> None:
    assert sec.key_matches("k", "k") is True
    assert sec.key_matches("k2", "k") is False
    assert sec.key_matches(None, "k") is False


def test_check_bind_allowed() -> None:
    sec.check_bind_allowed("127.0.0.1", None)
    sec.check_bind_allowed("0.0.0.0", "key")
    with pytest.raises(ValueError, match="OPENDROP_API_KEY"):
        sec.check_bind_allowed("0.0.0.0", None)
    with pytest.raises(ValueError):
        sec.check_bind_allowed("192.168.1.5", "")


def test_public_paths_are_minimal() -> None:
    assert sec.PUBLIC_PATHS == frozenset({"/health", "/", "/ui", "/favicon.ico"})


# ---------------------------------------------------------------------------
# App-level behaviour
# ---------------------------------------------------------------------------


def _deps():
    reg = AsyncMock()
    reg.list_models = AsyncMock(return_value=[])
    reg.get_model = AsyncMock(return_value=None)
    mgr = MagicMock()
    mgr.running_models.return_value = {}
    return mgr, reg


@pytest.fixture
def open_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(sec.API_KEY_ENV, raising=False)
    monkeypatch.delenv(sec.CORS_ORIGINS_ENV, raising=False)
    mgr, reg = _deps()
    with TestClient(create_app(manager=mgr, registry=reg)) as c:
        yield c


@pytest.fixture
def keyed_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(sec.CORS_ORIGINS_ENV, raising=False)
    mgr, reg = _deps()
    with TestClient(create_app(manager=mgr, registry=reg, api_key="s3cret")) as c:
        yield c


def test_foreign_origin_cannot_pull(open_client: TestClient, monkeypatch) -> None:
    called = MagicMock()
    monkeypatch.setattr("opendrop.inference.server.Orchestrator", called)
    r = open_client.post(
        "/v1/pull",
        content='{"source": "org/huge-model"}',
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
    )
    assert r.status_code == 403
    called.assert_not_called()


def test_foreign_origin_refused_on_get(open_client: TestClient) -> None:
    r = open_client.get("/v1/models", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_pull_requires_json_content_type(open_client: TestClient) -> None:
    r = open_client.post(
        "/v1/pull", content='{"source": "x"}', headers={"Content-Type": "text/plain"}
    )
    assert r.status_code == 415


def test_pull_accepts_json_with_charset(open_client: TestClient) -> None:
    # Reaches the handler (empty source -> 422), so the content-type gate passed.
    r = open_client.post(
        "/v1/pull",
        content='{"source": ""}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert r.status_code == 422


def test_loopback_origin_allowed_without_key(open_client: TestClient) -> None:
    r = open_client.get("/v1/models", headers={"Origin": "http://127.0.0.1:11400"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://127.0.0.1:11400"


def test_no_wildcard_cors(open_client: TestClient) -> None:
    r = open_client.options(
        "/v1/pull",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") not in ("*", "https://evil.example")


def test_preflight_from_loopback_allowed(open_client: TestClient) -> None:
    r = open_client.options(
        "/v1/pull",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_extra_origin_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sec.API_KEY_ENV, raising=False)
    monkeypatch.setenv(sec.CORS_ORIGINS_ENV, "https://app.example")
    mgr, reg = _deps()
    with TestClient(create_app(manager=mgr, registry=reg)) as c:
        assert c.get("/v1/models", headers={"Origin": "https://app.example"}).status_code == 200
        assert c.get("/v1/models", headers={"Origin": "https://x.example"}).status_code == 403


def test_api_key_required_when_set(keyed_client: TestClient) -> None:
    r = keyed_client.get("/v1/models")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert (
        keyed_client.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    )
    assert (
        keyed_client.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code
        == 200
    )
    assert keyed_client.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_api_key_blocks_pull(keyed_client: TestClient) -> None:
    r = keyed_client.post("/v1/pull", json={"source": "org/model"})
    assert r.status_code == 401


def test_health_public_with_key(keyed_client: TestClient) -> None:
    assert keyed_client.get("/health").status_code == 200


def test_preflight_not_blocked_by_key(keyed_client: TestClient) -> None:
    r = keyed_client.options(
        "/v1/models",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 200


def test_api_key_from_env_used_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sec.API_KEY_ENV, "envkey")
    mgr, reg = _deps()
    with TestClient(create_app(manager=mgr, registry=reg)) as c:
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer envkey"}).status_code == 200


def test_cors_disabled_still_enforces_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sec.API_KEY_ENV, raising=False)
    mgr, reg = _deps()
    with TestClient(create_app(manager=mgr, registry=reg, allow_cors=False)) as c:
        r = c.get("/v1/models", headers={"Origin": "http://localhost:3000"})
        assert r.status_code == 200
        assert "access-control-allow-origin" not in r.headers
        assert c.get("/v1/models", headers={"Origin": "https://evil.example"}).status_code == 403


# ---------------------------------------------------------------------------
# CLI: refuse public bind without a key
# ---------------------------------------------------------------------------


def test_serve_refuses_public_bind_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import uvicorn
    from click.testing import CliRunner

    import opendrop.cli as cli_mod
    import opendrop.config as config_mod

    monkeypatch.delenv(sec.API_KEY_ENV, raising=False)
    cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=11400, cors=True))
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    ran = MagicMock()
    monkeypatch.setattr(uvicorn, "run", ran)
    result = CliRunner().invoke(cli_mod.main, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 2
    assert "Refusing to bind" in result.output
    ran.assert_not_called()


# ---------------------------------------------------------------------------
# Fine-tuning: trust_remote_code is opt-in
# ---------------------------------------------------------------------------


class _StopError(Exception):
    pass


def _fake_hf(monkeypatch: pytest.MonkeyPatch, seen: dict) -> None:
    torch = types.ModuleType("torch")
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    datasets = types.ModuleType("datasets")
    datasets.Dataset = object  # type: ignore[attr-defined]
    transformers = types.ModuleType("transformers")

    class _Tok:
        @staticmethod
        def from_pretrained(model_id, **kw):
            seen["tokenizer"] = kw
            raise _StopError

    for name in (
        "AutoModelForCausalLM",
        "DataCollatorForLanguageModeling",
        "Trainer",
        "TrainingArguments",
    ):
        setattr(transformers, name, object)
    transformers.AutoTokenizer = _Tok  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_trust_remote_code_defaults_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert ft.TrainingConfig().trust_remote_code is False
    seen: dict = {}
    _fake_hf(monkeypatch, seen)
    monkeypatch.setattr(ft, "load_dataset", lambda _: [])
    with pytest.raises(_StopError):
        ft.fine_tune(
            "evil/repo", "d", tmp_path, cfg=ft.TrainingConfig(method="full"), produce_gguf=False
        )
    assert seen["tokenizer"]["trust_remote_code"] is False


def test_trust_remote_code_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    _fake_hf(monkeypatch, seen)
    monkeypatch.setattr(ft, "load_dataset", lambda _: [])
    printed: list[str] = []
    monkeypatch.setattr(ft.console, "print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    cfg = ft.TrainingConfig(method="full", trust_remote_code=True)
    with pytest.raises(_StopError):
        ft.fine_tune("org/repo", "d", tmp_path, cfg=cfg, produce_gguf=False)
    assert seen["tokenizer"]["trust_remote_code"] is True
    assert any("trust_remote_code is enabled" in p for p in printed)


def test_cli_trust_remote_code_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from click.testing import CliRunner

    import opendrop.cli as cli_mod
    import opendrop.config as config_mod
    import opendrop.core.registry as reg_mod

    cfg = SimpleNamespace(registry_db=lambda: tmp_path / "r.db", adapters_dir=lambda: tmp_path)
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(reg_mod, "Registry", lambda _db: SimpleNamespace(get_model=lambda _m: None))
    got: list = []

    def _fake(**kw):
        got.append(kw["cfg"].trust_remote_code)
        raise RuntimeError("stop")

    monkeypatch.setattr(ft, "fine_tune", _fake)
    runner = CliRunner()
    runner.invoke(cli_mod.main, ["fine-tune", "org/m", "--data", "d.jsonl"])
    runner.invoke(cli_mod.main, ["fine-tune", "org/m", "--data", "d.jsonl", "--trust-remote-code"])
    assert got == [False, True]
