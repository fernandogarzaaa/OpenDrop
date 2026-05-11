"""Model URL resolver for OpenDrop.

Supports:
  - Full HuggingFace model page:  https://huggingface.co/org/model
  - HF model ID:                  org/model
  - Direct GGUF URL:              https://huggingface.co/org/model/resolve/main/file.gguf
  - Local GGUF file:              /path/to/model.gguf
  - Local SafeTensors directory:  /path/to/model/

Resolution produces a :class:`ModelSpec` with all the information needed
to download and configure a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

HF_API = "https://huggingface.co/api"
HF_BASE = "https://huggingface.co"

# Known open / permissive licenses — anything else gets a warning
_OPEN_LICENSES = {
    "apache-2.0",
    "mit",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "gemma",
    "mistral",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "openrail",
    "openrail++",
    "wtfpl",
    "unlicense",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
}

_RESTRICTED_LICENSES = {
    "gpl-3.0",
    "gpl-2.0",
    "agpl-3.0",
}


@dataclass
class FileVariant:
    """A single downloadable file variant for a model."""

    filename: str
    url: str
    size_bytes: int = 0
    sha256: str = ""
    is_gguf: bool = False
    quant_label: str = ""  # e.g. "Q4_K_M" parsed from filename


@dataclass
class ModelSpec:
    """Everything OpenDrop knows about a model before downloading."""

    # Source
    source_url: str  # original input
    model_id: str  # "org/name"
    is_local: bool = False

    # Metadata
    model_name: str = ""  # display name
    architecture: str = ""  # e.g. "llama", "mistral", "qwen2"
    params_b: float = 0.0  # parameter count in billions
    license_id: str = ""
    license_ok: bool = True
    license_warning: str = ""
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str = ""  # "text-generation", "fill-mask", …

    # Download options
    variants: list[FileVariant] = field(default_factory=list)
    # Direct single-file download (set when URL points to a specific file)
    direct_file: FileVariant | None = None

    # Local paths (set when source is local)
    local_path: Path | None = None

    def best_gguf_variants(self) -> list[FileVariant]:
        """Return all GGUF variants, sorted by file size descending."""
        return sorted(
            (v for v in self.variants if v.is_gguf),
            key=lambda v: v.size_bytes,
            reverse=True,
        )

    def has_gguf(self) -> bool:
        return bool(self.direct_file and self.direct_file.is_gguf) or any(
            v.is_gguf for v in self.variants
        )


@dataclass
class ModelSearchResult:
    """A Hugging Face model search hit for CLI rendering."""

    model_id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    license_id: str = ""
    last_modified: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_quant_from_filename(name: str) -> str:
    """Try to extract a quant label from a GGUF filename."""
    patterns = [
        r"\.(Q\d+_K_[MSL])",
        r"\.(Q\d+_K)",
        r"\.(Q\d+_\d+)",
        r"\.(Q[0-9]+)",
        r"\.(IQ\d+_\w+)",
        r"\.(fp16)",
        r"\.(f16)",
        r"\.(f32)",
    ]
    for pat in patterns:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def _check_license(license_id: str) -> tuple[bool, str]:
    """Return (ok, warning_message)."""
    lid = license_id.lower().strip()
    if lid in _OPEN_LICENSES:
        return True, ""
    if lid in _RESTRICTED_LICENSES:
        return True, (
            f"License '{license_id}' is {lid.upper()} — copyleft terms may apply to "
            "derivative works. Verify compliance with your use case."
        )
    if lid in ("other", "", "custom"):
        return True, (
            f"License '{license_id or 'unspecified'}' — verify the model card manually "
            "before commercial or redistributed use."
        )
    return True, f"Unknown license '{license_id}' — review the model card before use."


def _is_hf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in ("huggingface.co", "hf.co")


def _extract_hf_model_id(url: str) -> str | None:
    """Extract 'org/model' from a HF URL or return None."""
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    # Covers /org/model, /org/model/blob/…, /org/model/resolve/…
    parts = path.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def _is_direct_file_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.path.endswith(".gguf")
        or parsed.path.endswith(".bin")
        or parsed.path.endswith(".safetensors")
        or "/resolve/" in parsed.path
    )


def _build_direct_variant(url: str) -> FileVariant:
    filename = url.split("/")[-1].split("?")[0]
    is_gguf = filename.endswith(".gguf")
    return FileVariant(
        filename=filename,
        url=url,
        is_gguf=is_gguf,
        quant_label=_parse_quant_from_filename(filename),
    )


def _params_from_name(name: str) -> float:
    """Heuristic: extract parameter count from model name, e.g. 'Llama-3-8B' → 8.0."""
    m = re.search(r"(\d+\.?\d*)[Bb]", name)
    if m:
        return float(m.group(1))
    # Fallback: 'Llama-3-70B-Instruct' etc
    m = re.search(r"-(\d+)[Bb]", name)
    if m:
        return float(m.group(1))
    return 0.0


# ---------------------------------------------------------------------------
# HuggingFace API fetch helpers (synchronous for CLI simplicity)
# ---------------------------------------------------------------------------


def _hf_model_info(model_id: str, token: str | None = None) -> dict:
    """Fetch model metadata from the HF API."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{HF_API}/models/{model_id}"
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()


def _hf_model_files(model_id: str, token: str | None = None) -> list[dict]:
    """Fetch the file listing for a HF model repo."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{HF_API}/models/{model_id}/tree/main"
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        r = client.get(url, headers=headers)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()


def _build_variants_from_tree(model_id: str, tree: list[dict]) -> list[FileVariant]:
    variants: list[FileVariant] = []
    for entry in tree:
        if entry.get("type") != "file":
            continue
        name = entry.get("path", "")
        size = entry.get("size", 0)
        is_gguf = name.endswith(".gguf")
        is_safetensors = name.endswith(".safetensors")
        if not (is_gguf or is_safetensors):
            continue
        url = f"{HF_BASE}/{model_id}/resolve/main/{name}"
        variants.append(
            FileVariant(
                filename=name,
                url=url,
                size_bytes=size,
                is_gguf=is_gguf,
                quant_label=_parse_quant_from_filename(name) if is_gguf else "",
            )
        )
    return variants


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(source: str, token: str | None = None) -> ModelSpec:
    """Resolve *source* (URL, HF model ID, or local path) to a :class:`ModelSpec`.

    Args:
        source: A HuggingFace URL, 'org/model' ID, or local file/directory path.
        token:  Optional HuggingFace token for private repos.

    Returns:
        A ModelSpec ready for the download stage.

    Raises:
        ValueError:    When the source cannot be understood.
        httpx.HTTPError: When the HF API is unreachable.
    """
    # --- Local path ---------------------------------------------------------
    local = Path(source).expanduser()
    if local.exists():
        return _resolve_local(local)

    # --- Direct GGUF/SafeTensors URL ----------------------------------------
    if _is_hf_url(source) and _is_direct_file_url(source):
        model_id = _extract_hf_model_id(source) or "unknown/unknown"
        variant = _build_direct_variant(source)
        spec = ModelSpec(
            source_url=source,
            model_id=model_id,
            model_name=model_id.split("/")[-1],
            direct_file=variant,
        )
        spec.params_b = _params_from_name(model_id)
        _enrich_from_hf(spec, model_id, token)
        return spec

    # --- HuggingFace model page URL or bare model ID ------------------------
    if _is_hf_url(source):
        hf_model_id = _extract_hf_model_id(source)
        if not hf_model_id:
            raise ValueError(f"Cannot extract model ID from URL: {source}")
        model_id = hf_model_id
    elif "/" in source and not source.startswith("http"):
        # bare 'org/model' ID
        model_id = source
    else:
        raise ValueError(
            f"Cannot resolve '{source}'. Provide a HuggingFace URL, 'org/model' ID, "
            "or a local file/directory path."
        )

    spec = ModelSpec(source_url=source, model_id=model_id, model_name=model_id.split("/")[-1])
    _enrich_from_hf(spec, model_id, token)
    return spec


def search_models(
    query: str,
    limit: int = 10,
    token: str | None = None,
) -> list[ModelSearchResult]:
    """Search Hugging Face models."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params: dict[str, str | int] = {"search": query, "limit": max(1, min(limit, 50))}

    with httpx.Client(follow_redirects=True, timeout=30) as client:
        r = client.get(f"{HF_API}/models", headers=headers, params=params)
        r.raise_for_status()
        payload = r.json()

    results: list[ModelSearchResult] = []
    for item in payload:
        tags = item.get("tags") or []
        license_id = ""
        card = item.get("cardData")
        if isinstance(card, dict):
            license_id = card.get("license") or ""
        if not license_id:
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("license:"):
                    license_id = tag.split(":", 1)[1]
                    break

        results.append(
            ModelSearchResult(
                model_id=item.get("id") or "",
                downloads=int(item.get("downloads") or 0),
                likes=int(item.get("likes") or 0),
                pipeline_tag=item.get("pipeline_tag") or "",
                license_id=license_id,
                last_modified=item.get("lastModified") or "",
            )
        )

    return [r for r in results if r.model_id]


def _resolve_local(path: Path) -> ModelSpec:
    spec = ModelSpec(
        source_url=str(path),
        model_id=f"local/{path.name}",
        model_name=path.name,
        is_local=True,
        local_path=path,
    )
    if path.is_file() and path.suffix == ".gguf":
        spec.direct_file = FileVariant(
            filename=path.name,
            url=str(path),
            size_bytes=path.stat().st_size,
            is_gguf=True,
            quant_label=_parse_quant_from_filename(path.name),
        )
    spec.params_b = _params_from_name(path.name)
    return spec


def _enrich_from_hf(spec: ModelSpec, model_id: str, token: str | None) -> None:
    """Fetch HF metadata and file tree, populating spec in-place."""
    try:
        info = _hf_model_info(model_id, token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise ValueError(f"Model '{model_id}' not found on HuggingFace.") from exc
        raise
    except httpx.RequestError as exc:
        raise ConnectionError(
            f"Cannot reach HuggingFace API: {exc}. Check your network connection."
        ) from exc

    spec.architecture = (info.get("config") or {}).get("model_type", "")
    spec.tags = info.get("tags") or []
    spec.pipeline_tag = info.get("pipeline_tag") or ""
    license_id = info.get("cardData", {}).get("license", "")
    if not license_id:
        for tag in spec.tags:
            if tag.startswith("license:"):
                license_id = tag.split(":", 1)[1]
                break
    spec.license_id = license_id
    spec.license_ok, spec.license_warning = _check_license(license_id)

    if not spec.params_b:
        spec.params_b = _params_from_name(model_id)
        if not spec.params_b:
            # Try safetensors_info
            si = info.get("safetensors") or {}
            total = sum(
                si.get("total", {}).values()
                if isinstance(si.get("total"), dict)
                else [si.get("total", 0)]
            )
            if total:
                spec.params_b = total / 1e9

    # File tree
    if not spec.direct_file:
        try:
            tree = _hf_model_files(model_id, token)
            spec.variants = _build_variants_from_tree(model_id, tree)
        except Exception:
            pass  # Non-fatal: tree may be unavailable for gated repos
