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


def load_config(path: Path | None = None) -> Config:
    """Load and return the merged Config object."""
    if path is None:
        path = Path(user_config_dir(_APP)) / "config.toml"

    data = _deep_merge(_DEFAULTS, {})
    if path.exists():
        with open(path, "rb") as fh:
            user_data = tomllib.load(fh)
        data = _deep_merge(data, user_data)

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
