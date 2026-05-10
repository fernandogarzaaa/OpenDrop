"""Tests for opendrop.training.dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from opendrop.training.dataset import (
    DatasetError,
    _is_alpaca,
    _is_sharegpt,
    _normalize_alpaca,
    _normalize_prompt_completion,
    _normalize_sharegpt,
    dataset_stats,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Normalizer unit tests
# ---------------------------------------------------------------------------

class TestNormalizers:
    def test_alpaca_detection(self):
        sample = {"instruction": "Translate", "output": "Translated"}
        assert _is_alpaca(sample)
        assert not _is_sharegpt(sample)

    def test_sharegpt_detection(self):
        sample = {"conversations": [{"from": "human", "value": "Hi"}]}
        assert _is_sharegpt(sample)
        assert not _is_alpaca(sample)

    def test_normalize_alpaca(self):
        sample = {"instruction": "Say hello", "input": "", "output": "Hello!"}
        result = _normalize_alpaca(sample)
        assert "messages" in result
        msgs = result["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hello!"

    def test_normalize_alpaca_with_input(self):
        sample = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        result = _normalize_alpaca(sample)
        assert "Hello" in result["messages"][0]["content"]

    def test_normalize_sharegpt(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Hi there"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = _normalize_sharegpt(sample)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_normalize_prompt_completion(self):
        sample = {"prompt": "Q: What?", "completion": "A: This."}
        result = _normalize_prompt_completion(sample)
        assert result["messages"][0]["content"] == "Q: What?"
        assert result["messages"][1]["content"] == "A: This."


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

class TestLoadJSONL:
    def test_load_alpaca_jsonl(self, tmp_path: Path):
        data = [
            {"instruction": "Summarize", "input": "", "output": "Summary"},
            {"instruction": "Translate", "input": "Hello", "output": "Hola"},
        ]
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(json.dumps(d) for d in data))
        result = load_dataset(str(f))
        assert len(result) == 2
        assert "messages" in result[0]

    def test_load_sharegpt_jsonl(self, tmp_path: Path):
        data = [
            {"conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ]}
        ]
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(data[0]))
        result = load_dataset(str(f))
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "user"

    def test_load_prompt_completion_jsonl(self, tmp_path: Path):
        data = [{"prompt": "A?", "completion": "B."}]
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(data[0]))
        result = load_dataset(str(f))
        assert result[0]["messages"][0]["content"] == "A?"

    def test_invalid_json_raises(self, tmp_path: Path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"ok": true}\nNOT JSON\n')
        with pytest.raises(DatasetError):
            load_dataset(str(f))

    def test_empty_lines_skipped(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text('\n{"instruction":"Do","output":"Done"}\n\n')
        result = load_dataset(str(f))
        assert len(result) == 1

    def test_max_samples_respected(self, tmp_path: Path):
        data = [{"instruction": f"Task {i}", "output": f"Result {i}"} for i in range(20)]
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(json.dumps(d) for d in data))
        result = load_dataset(str(f), max_samples=5)
        assert len(result) == 5


class TestLoadCSV:
    def test_load_instruction_response_csv(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        with open(f, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["instruction", "response"])
            writer.writeheader()
            writer.writerow({"instruction": "Say hi", "response": "Hi!"})
        result = load_dataset(str(f))
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "user"

    def test_load_prompt_completion_csv(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        with open(f, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["prompt", "completion"])
            writer.writeheader()
            writer.writerow({"prompt": "Q", "completion": "A"})
        result = load_dataset(str(f))
        assert len(result) == 1


class TestLoadText:
    def test_load_text_splits_paragraphs(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
        result = load_dataset(str(f))
        assert len(result) == 3
        assert "text" in result[0]


class TestDatasetStats:
    def test_stats_all_chat(self, tmp_path: Path):
        data = [{"messages": [{"role": "user", "content": "Hi"},
                               {"role": "assistant", "content": "Hello"}]}
                for _ in range(5)]
        stats = dataset_stats(data)
        assert stats["total"] == 5
        assert stats["chat_format"] == 5
        assert stats["text_format"] == 0
        assert stats["avg_turns"] == 2.0

    def test_stats_mixed(self):
        data = [
            {"messages": [{"role": "user", "content": "Hi"}]},
            {"text": "Some raw text"},
        ]
        stats = dataset_stats(data)
        assert stats["total"] == 2
        assert stats["chat_format"] == 1
        assert stats["text_format"] == 1
