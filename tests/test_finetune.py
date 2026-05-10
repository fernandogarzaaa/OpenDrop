"""Tests for opendrop.training.finetune."""

from __future__ import annotations

from pathlib import Path

import pytest

from opendrop.training import finetune as ft


def test_fine_tune_lora_propagates_training_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ft, "load_dataset", lambda _: [{"prompt": "hello", "completion": "world"}])
    monkeypatch.setattr(
        ft,
        "_train_lora_peft",
        lambda *args, **kwargs: (adapter_dir, 1.234),
    )

    result = ft.fine_tune(
        model_id="org/model",
        dataset_source="dummy",
        output_dir=tmp_path,
        cfg=ft.TrainingConfig(method="lora"),
        produce_gguf=False,
    )

    assert result.adapter_dir == adapter_dir
    assert result.final_loss == pytest.approx(1.234)


def test_fine_tune_warns_when_gguf_output_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []

    monkeypatch.setattr(ft, "load_dataset", lambda _: [{"prompt": "hello", "completion": "world"}])
    monkeypatch.setattr(ft, "_train_lora_peft", lambda *args, **kwargs: (adapter_dir, 0.9))
    monkeypatch.setattr(ft, "_merge_lora_then_convert", lambda **kwargs: None)
    monkeypatch.setattr(
        ft.console,
        "print",
        lambda *args, **kwargs: messages.append(" ".join(str(a) for a in args)),
    )

    result = ft.fine_tune(
        model_id="org/model",
        dataset_source="dummy",
        output_dir=tmp_path,
        cfg=ft.TrainingConfig(method="lora"),
        produce_gguf=True,
    )

    assert result.merged_gguf is None
    assert any("no .gguf output file was found" in m for m in messages)
