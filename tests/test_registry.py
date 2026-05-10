"""Tests for opendrop.core.registry."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from opendrop.core.registry import AdapterRecord, ModelRecord, Registry


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_registry.db"


@pytest.fixture
def registry(db_path: Path) -> Registry:
    return Registry(db_path)


def _make_record(suffix: str = "") -> ModelRecord:
    return ModelRecord(
        id=f"llama-3-8b-q4-km{suffix}",
        model_id=f"meta-llama/Meta-Llama-3-8B{suffix}",
        source_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B",
        display_name=f"llama-3-8b-q4-km{suffix}",
        architecture="llama",
        params_b=8.0,
        quant="Q4_K_M",
        format="gguf",
        path="/tmp/llama.gguf",
        size_bytes=4_500_000_000,
        license_id="llama3",
        license_warning="",
        tags=["text-generation"],
        pipeline_tag="text-generation",
        added_at=datetime.now(timezone.utc).isoformat(),
        last_used=None,
        server_port=None,
        extra={},
    )


class TestRegistry:
    def test_add_and_get_model(self, registry: Registry):
        rec = _make_record()
        registry.add_model(rec)
        fetched = registry.get_model(rec.id)
        assert fetched is not None
        assert fetched.id == rec.id
        assert fetched.params_b == 8.0

    def test_get_by_display_name(self, registry: Registry):
        rec = _make_record()
        registry.add_model(rec)
        fetched = registry.get_model(rec.display_name)
        assert fetched is not None

    def test_get_nonexistent_returns_none(self, registry: Registry):
        assert registry.get_model("nonexistent-model") is None

    def test_list_models_empty(self, registry: Registry):
        assert registry.list_models() == []

    def test_list_models(self, registry: Registry):
        registry.add_model(_make_record("-a"))
        registry.add_model(_make_record("-b"))
        records = registry.list_models()
        assert len(records) == 2

    def test_remove_model(self, registry: Registry):
        rec = _make_record()
        registry.add_model(rec)
        assert registry.remove_model(rec.id) is True
        assert registry.get_model(rec.id) is None

    def test_remove_nonexistent_returns_false(self, registry: Registry):
        assert registry.remove_model("ghost") is False

    def test_touch_model(self, registry: Registry):
        rec = _make_record()
        registry.add_model(rec)
        registry.touch_model(rec.id)
        fetched = registry.get_model(rec.id)
        assert fetched.last_used is not None

    def test_set_port(self, registry: Registry):
        rec = _make_record()
        registry.add_model(rec)
        registry.set_port(rec.id, 11401)
        fetched = registry.get_model(rec.id)
        assert fetched.server_port == 11401

    def test_tags_serialized(self, registry: Registry):
        rec = _make_record()
        rec.tags = ["text-generation", "llama", "gguf"]
        registry.add_model(rec)
        fetched = registry.get_model(rec.id)
        assert fetched.tags == ["text-generation", "llama", "gguf"]

    def test_extra_dict_serialized(self, registry: Registry):
        rec = _make_record()
        rec.extra = {"custom_key": 42, "nested": {"a": 1}}
        registry.add_model(rec)
        fetched = registry.get_model(rec.id)
        assert fetched.extra["custom_key"] == 42

    def test_upsert_replaces_existing(self, registry: Registry):
        rec = _make_record()
        registry.add_model(rec)
        rec.params_b = 13.0
        registry.add_model(rec)
        fetched = registry.get_model(rec.id)
        assert fetched.params_b == 13.0

    # --- Adapters ---

    def test_add_and_list_adapter(self, registry: Registry):
        model = _make_record()
        registry.add_model(model)
        adapter = AdapterRecord(
            id="adapter-001",
            model_id=model.id,
            name="my-lora",
            method="lora",
            path="/tmp/adapter",
            dataset_url="alpaca",
            added_at=datetime.now(timezone.utc).isoformat(),
            extra={},
        )
        registry.add_adapter(adapter)
        adapters = registry.list_adapters(model.id)
        assert len(adapters) == 1
        assert adapters[0].name == "my-lora"

    def test_remove_adapter(self, registry: Registry):
        model = _make_record()
        registry.add_model(model)
        adapter = AdapterRecord(
            id="adapter-002",
            model_id=model.id,
            name="my-lora",
            method="lora",
            path="/tmp/adapter",
            dataset_url="",
            added_at=datetime.now(timezone.utc).isoformat(),
            extra={},
        )
        registry.add_adapter(adapter)
        assert registry.remove_adapter("adapter-002") is True
        assert registry.list_adapters(model.id) == []


class TestModelRecord:
    def test_size_human_gb(self):
        rec = _make_record()
        rec.size_bytes = 4_500_000_000
        assert "GB" in rec.size_human()

    def test_size_human_mb(self):
        rec = _make_record()
        rec.size_bytes = 500_000_000
        assert "MB" in rec.size_human()

    def test_path_obj(self):
        rec = _make_record()
        rec.path = "/tmp/model.gguf"
        assert rec.path_obj() == Path("/tmp/model.gguf")
