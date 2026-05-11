"""Configuration management for OpenDrop.

Reads from ~/.config/opendrop/config.toml (XDG) and provides typed defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir, user_data_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-reuse-declaration]

_APP = "opendrop"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 11400,
        "cors": True,
    },
    "storage": {
        "models_dir": str(Path(user_data_dir(_APP)) / "models"),
        "registry_db": str(Path(user_data_dir(_APP)) / "registry.db"),
        "llamacpp_dir": "",  # empty → search PATH + common locations
    },
    "inference": {
        "context_size": 8192,
        "disk_kv_cache": True,
        "gpu_layers": -1,  # -1 = all to GPU
        "parallel": 1,  # concurrent slots per model
        "flash_attn": True,
    },
    "training": {
        "default_method": "lora",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "warmup_ratio": 0.03,
        "max_seq_length": 2048,
        "output_dir": str(Path(user_data_dir(_APP)) / "adapters"),
    },
}


class _Section:
    """Dot-access wrapper for a config section dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Section({self._data!r})"


class Config:
    """Merged configuration object (defaults + file overrides)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def server(self) -> _Section:
        return _Section(self._data["server"])

    @property
    def storage(self) -> _Section:
        return _Section(self._data["storage"])

    @property
    def inference(self) -> _Section:
        return _Section(self._data["inference"])

    @property
    def training(self) -> _Section:
        return _Section(self._data["training"])

    def models_dir(self) -> Path:
        p = Path(self._data["storage"]["models_dir"]).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def registry_db(self) -> Path:
        p = Path(self._data["storage"]["registry_db"]).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def adapters_dir(self) -> Path:
        p = Path(self._data["training"]["output_dir"]).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _ensure_int_in_range(
    section: str,
    key: str,
    value: Any,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int):
        raise ValueError(f"config [{section}].{key} must be an integer")
    if value < minimum:
        raise ValueError(f"config [{section}].{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"config [{section}].{key} must be <= {maximum}")
    return value


def _ensure_bool(section: str, key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"config [{section}].{key} must be a boolean")
    return value


def _validate_config_data(data: dict[str, Any]) -> dict[str, Any]:
    server = data.get("server", {})
    storage = data.get("storage", {})
    inference = data.get("inference", {})
    training = data.get("training", {})

    host = server.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("config [server].host must be a non-empty string")
    _ensure_int_in_range("server", "port", server.get("port"), 1, 65535)
    _ensure_bool("server", "cors", server.get("cors"))

    for section_name, section_data, key in (
        ("storage", storage, "models_dir"),
        ("storage", storage, "registry_db"),
        ("training", training, "output_dir"),
    ):
        raw_value = section_data.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"config [{section_name}].{key} must be a non-empty string path")

    _ensure_int_in_range("inference", "context_size", inference.get("context_size"), 1)
    _ensure_int_in_range("inference", "parallel", inference.get("parallel"), 1)
    gpu_layers = inference.get("gpu_layers")
    if not isinstance(gpu_layers, int):
        raise ValueError("config [inference].gpu_layers must be an integer")
    if gpu_layers < -1:
        raise ValueError("config [inference].gpu_layers must be >= -1")
    _ensure_bool("inference", "disk_kv_cache", inference.get("disk_kv_cache"))
    _ensure_bool("inference", "flash_attn", inference.get("flash_attn"))

    _ensure_int_in_range("training", "lora_rank", training.get("lora_rank"), 1)
    _ensure_int_in_range("training", "lora_alpha", training.get("lora_alpha"), 1)
    _ensure_int_in_range("training", "batch_size", training.get("batch_size"), 1)
    _ensure_int_in_range(
        "training",
        "gradient_accumulation",
        training.get("gradient_accumulation"),
        1,
    )
    _ensure_int_in_range("training", "max_seq_length", training.get("max_seq_length"), 1)

    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("config [training].learning_rate must be > 0")
    warmup_ratio = training.get("warmup_ratio")
    if not isinstance(warmup_ratio, (int, float)) or not 0 <= warmup_ratio <= 1:
        raise ValueError("config [training].warmup_ratio must be between 0 and 1")

    default_method = training.get("default_method")
    if default_method not in {"lora", "qlora", "full", "mlx"}:
        raise ValueError(
            "config [training].default_method must be one of: lora, qlora, full, mlx"
        )

    return data


def load_config(path: Path | None = None) -> Config:
    """Load and return the merged Config object."""
    if path is None:
        path = Path(user_config_dir(_APP)) / "config.toml"

    data = _deep_merge(_DEFAULTS, {})
    if path.exists():
        with open(path, "rb") as fh:
            user_data = tomllib.load(fh)
        data = _deep_merge(data, user_data)
    data = _validate_config_data(data)

    return Config(data)


# Module-level singleton — lazy so tests can patch it
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Force reload — used in tests."""
    global _config
    _config = None
