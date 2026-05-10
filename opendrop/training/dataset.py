"""Dataset loader for OpenDrop fine-tuning pipeline.

Supports:
  - JSONL  (instruction/response, prompt/completion, messages/conversations)
  - CSV    (with header detection for instruction/response columns)
  - HuggingFace dataset ID  (e.g. 'tatsu-lab/alpaca')
  - Raw text files          (for continued pre-training)
  - Alpaca format           ({"instruction": …, "input": …, "output": …})
  - ShareGPT format         ({"conversations": [{"from": "human", "value": …}, …]})

Output is always a list of dicts with 'prompt' and 'completion' keys,
or 'messages' key for chat-format datasets.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Sample = dict[str, Any]
Dataset = list[Sample]


class DatasetError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _is_alpaca(sample: dict) -> bool:
    return "instruction" in sample and "output" in sample


def _is_sharegpt(sample: dict) -> bool:
    return "conversations" in sample and isinstance(sample["conversations"], list)


def _is_messages(sample: dict) -> bool:
    return "messages" in sample and isinstance(sample["messages"], list)


def _is_prompt_completion(sample: dict) -> bool:
    return "prompt" in sample and "completion" in sample


def _is_instruction_response(sample: dict) -> bool:
    return "instruction" in sample and "response" in sample


# ---------------------------------------------------------------------------
# Normalizers → standard {messages: [...]} format
# ---------------------------------------------------------------------------


def _normalize_alpaca(sample: dict) -> Sample:
    instruction = sample["instruction"].strip()
    input_text = sample.get("input", "").strip()
    output = sample["output"].strip()
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def _normalize_sharegpt(sample: dict) -> Sample:
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in sample["conversations"]:
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        messages.append({"role": role, "content": turn.get("value", "")})
    return {"messages": messages}


def _normalize_messages(sample: dict) -> Sample:
    return {"messages": sample["messages"]}


def _normalize_prompt_completion(sample: dict) -> Sample:
    return {
        "messages": [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["completion"]},
        ]
    }


def _normalize_instruction_response(sample: dict) -> Sample:
    return {
        "messages": [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["response"]},
        ]
    }


def _normalize_raw_text(text: str) -> Sample:
    return {"text": text}


def _normalize_sample(sample: dict) -> Sample:
    if _is_sharegpt(sample):
        return _normalize_sharegpt(sample)
    if _is_messages(sample):
        return _normalize_messages(sample)
    if _is_alpaca(sample):
        return _normalize_alpaca(sample)
    if _is_prompt_completion(sample):
        return _normalize_prompt_completion(sample)
    if _is_instruction_response(sample):
        return _normalize_instruction_response(sample)
    # Pass through unknown formats
    return sample


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> Dataset:
    samples: Dataset = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON on line {i + 1} of {path}: {exc}") from exc
            samples.append(_normalize_sample(obj))
    return samples


def _load_csv(path: Path) -> Dataset:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise DatasetError(f"CSV file has no header: {path}")
        fields = set(reader.fieldnames)
        samples: Dataset = []
        for row in reader:
            if "instruction" in fields and "output" in fields:
                samples.append(_normalize_alpaca(dict(row)))
            elif "prompt" in fields and "completion" in fields:
                samples.append(_normalize_prompt_completion(dict(row)))
            elif "instruction" in fields and "response" in fields:
                samples.append(_normalize_instruction_response(dict(row)))
            else:
                # Treat as raw text using the first column
                first = reader.fieldnames[0]
                samples.append(_normalize_raw_text(row[first]))
    return samples


def _load_text(path: Path) -> Dataset:
    text = path.read_text(encoding="utf-8")
    # Split on double newlines or paragraph boundaries
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return [_normalize_raw_text(c) for c in chunks]


def _load_hf_dataset(dataset_id: str, split: str = "train") -> Dataset:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise DatasetError(
            "HuggingFace 'datasets' package not installed. Run: pip install opendrop[training]"
        ) from exc

    ds = load_dataset(dataset_id, split=split)
    return [_normalize_sample(dict(row)) for row in ds]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    source: str,
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    """Load a dataset from *source* and return normalized samples.

    Args:
        source:      File path (JSONL / CSV / .txt) or HF dataset ID.
        split:       HF dataset split (only used when source is an HF ID).
        max_samples: Cap the number of samples (None = no cap).

    Returns:
        List of normalized sample dicts with 'messages' or 'text' keys.
    """
    path = Path(source).expanduser()

    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".jsonl" or suffix == ".json":
            data = _load_jsonl(path)
        elif suffix == ".csv":
            data = _load_csv(path)
        elif suffix in (".txt", ".md"):
            data = _load_text(path)
        else:
            # Try JSONL first, fall back to text
            try:
                data = _load_jsonl(path)
            except DatasetError:
                data = _load_text(path)
    elif "/" in source or not path.suffix:
        # Treat as HF dataset ID
        data = _load_hf_dataset(source, split=split)
    else:
        raise DatasetError(
            f"Cannot load dataset from '{source}'. "
            "Provide a file path (JSONL/CSV/TXT) or a HuggingFace dataset ID."
        )

    if max_samples is not None:
        data = data[:max_samples]

    return data


def format_sample_for_training(sample: Sample, tokenizer: Any) -> str:
    """Convert a normalized sample to a training string using the tokenizer's chat template."""
    if "messages" in sample:
        try:
            return tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fall back to simple concatenation
            return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in sample["messages"])
    return sample.get("text", "")


def dataset_stats(data: Dataset) -> dict:
    """Return basic statistics about a loaded dataset."""
    n = len(data)
    chat = sum(1 for s in data if "messages" in s)
    text_only = n - chat
    avg_turns = sum(len(s["messages"]) for s in data if "messages" in s) / chat if chat else 0
    return {
        "total": n,
        "chat_format": chat,
        "text_format": text_only,
        "avg_turns": round(avg_turns, 2),
    }
