"""Orchestrator — ties resolver, hardware, quantizer, downloader, converter,
and registry together into the high-level `pull` and `prepare` operations.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from opendrop.config import get_config
from opendrop.core.converter import (
    convert_and_quantize,
    needs_conversion,
)
from opendrop.core.downloader import DownloadError, download
from opendrop.core.hardware import HardwareProfile, detect_hardware
from opendrop.core.quantizer import QuantSpec, select_quantization
from opendrop.core.registry import ModelRecord, Registry
from opendrop.core.resolver import FileVariant, ModelSpec, resolve

console = Console()


def _safe_id(name: str) -> str:
    """Turn a display name into a registry-safe ID."""
    return re.sub(r"[^a-z0-9_-]", "-", name.lower()).strip("-")


def _unique_id(base: str) -> str:
    return f"{_safe_id(base)}-{uuid.uuid4().hex[:6]}"


class Orchestrator:
    """High-level model lifecycle operations."""

    def __init__(
        self,
        registry: Registry | None = None,
        profile: HardwareProfile | None = None,
    ) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._registry = registry or Registry(cfg.registry_db())
        self._profile = profile or detect_hardware()

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def profile(self) -> HardwareProfile:
        return self._profile

    # ------------------------------------------------------------------
    # pull = resolve + download + convert + register
    # ------------------------------------------------------------------

    def pull(
        self,
        source: str,
        token: str | None = None,
        quant_override: str | None = None,
        force: bool = False,
    ) -> ModelRecord:
        """Full model pull pipeline.

        1. Resolve URL / model ID → ModelSpec
        2. Check license
        3. Select quantization
        4. Download the GGUF (or download + convert)
        5. Register in SQLite registry
        6. Return ModelRecord

        Args:
            source:         HF URL, 'org/model', or local path.
            token:          Optional HF auth token.
            quant_override: Force a specific quantization (bypasses auto).
            force:          Re-download even if already registered.

        Returns:
            The :class:`ModelRecord` added to the registry.
        """
        # --- Resolve --------------------------------------------------------
        console.print(f"[bold]Resolving[/bold] {source} …")
        spec = resolve(source, token=token)

        if spec.license_warning:
            console.print(f"[yellow]⚠ License: {spec.license_warning}[/yellow]")

        console.print(f"  Model  : [cyan]{spec.model_name}[/cyan]  ({spec.params_b:.1f}B params)")
        if spec.architecture:
            console.print(f"  Arch   : {spec.architecture}")
        if spec.pipeline_tag:
            console.print(f"  Task   : {spec.pipeline_tag}")

        # --- Quantization decision ------------------------------------------
        quant_spec = select_quantization(self._profile, spec.params_b, quant_override)
        console.print(
            f"  Quant  : [green]{quant_spec.name}[/green] ({quant_spec.quality})"
            f" — {quant_spec.description}"
        )

        # --- Check if already registered -------------------------------------
        existing = self._find_existing(spec, quant_spec)
        if existing and not force:
            console.print(f"[green]✓ Already in registry:[/green] {existing.display_name}")
            return existing

        # --- Download / convert -----------------------------------------------
        dest_dir = self._cfg.models_dir() / _safe_id(spec.model_name)
        gguf_path = self._acquire_gguf(spec, quant_spec, dest_dir, token)

        # --- Register ---------------------------------------------------------
        record = self._register(spec, quant_spec, gguf_path)
        console.print(
            f"[green]✓ Registered:[/green] [bold]{record.display_name}[/bold] "
            f"({record.size_human()})"
        )
        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_existing(self, spec: ModelSpec, quant: QuantSpec) -> ModelRecord | None:
        for rec in self._registry.list_models():
            if rec.model_id == spec.model_id and rec.quant == quant.name:
                if Path(rec.path).exists():
                    return rec
        return None

    def _acquire_gguf(
        self,
        spec: ModelSpec,
        quant: QuantSpec,
        dest_dir: Path,
        token: str | None,
    ) -> Path:
        """Return path to a ready-to-use GGUF file."""

        # Local path — maybe already GGUF, maybe needs conversion
        if spec.is_local and spec.local_path:
            if spec.local_path.suffix == ".gguf":
                return spec.local_path
            if needs_conversion(spec.local_path):
                return convert_and_quantize(spec.local_path, dest_dir, quant.name)
            raise ValueError(f"Cannot determine how to use {spec.local_path}")

        # Direct file URL (already a GGUF)
        if spec.direct_file and spec.direct_file.is_gguf:
            return download(
                spec.direct_file.url,
                dest_dir,
                filename=spec.direct_file.filename,
                token=token,
            )

        # Pick the GGUF variant that best matches the requested quant
        gguf_variants = spec.best_gguf_variants()
        if gguf_variants:
            selected = self._best_gguf_match(gguf_variants, quant)
            console.print(f"  File   : {selected.filename}")
            return download(selected.url, dest_dir, filename=selected.filename, token=token)

        # No GGUF available → download SafeTensors + convert
        safetensors = [v for v in spec.variants if v.filename.endswith(".safetensors")]
        if not safetensors:
            raise DownloadError(
                f"No GGUF or SafeTensors files found for model '{spec.model_id}'. "
                "The model may be gated — try providing a --token."
            )
        console.print("[yellow]No GGUF found — downloading SafeTensors for conversion…[/yellow]")
        for v in safetensors:
            download(v.url, dest_dir / "src", filename=v.filename, token=token)
        # Also grab config files
        config_files = [
            v
            for v in spec.variants
            if v.filename
            in ("config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        ]
        for v in config_files:
            download(v.url, dest_dir / "src", filename=v.filename, token=token)

        return convert_and_quantize(dest_dir / "src", dest_dir, quant.name)

    def _best_gguf_match(self, variants: list[FileVariant], quant: QuantSpec) -> FileVariant:
        """Find the variant whose quant_label best matches requested quant."""
        for v in variants:
            if v.quant_label.upper() == quant.name.upper():
                return v
        # Fall back: pick smallest that has a quant label (most compressed)
        labelled = [v for v in variants if v.quant_label]
        if labelled:
            return min(labelled, key=lambda v: v.size_bytes)
        return variants[0]

    def _register(self, spec: ModelSpec, quant: QuantSpec, gguf_path: Path) -> ModelRecord:
        display = f"{spec.model_name}-{quant.name}".lower()
        rec_id = _unique_id(display)
        # Ensure unique display name
        existing_names = {r.display_name for r in self._registry.list_models()}
        if display not in existing_names:
            rec_id = _safe_id(display)

        record = ModelRecord(
            id=rec_id,
            model_id=spec.model_id,
            source_url=spec.source_url,
            display_name=display,
            architecture=spec.architecture,
            params_b=spec.params_b,
            quant=quant.name,
            format="gguf",
            path=str(gguf_path),
            size_bytes=gguf_path.stat().st_size if gguf_path.exists() else 0,
            license_id=spec.license_id,
            license_warning=spec.license_warning,
            tags=spec.tags,
            pipeline_tag=spec.pipeline_tag,
            added_at=datetime.now(timezone.utc).isoformat(),
            last_used=None,
            server_port=None,
            extra={},
        )
        self._registry.add_model(record)
        return record

    def remove(self, model_id: str, delete_files: bool = True) -> bool:
        """Remove a model from registry and optionally delete its files."""
        rec = self._registry.get_model(model_id)
        if not rec:
            return False
        if delete_files:
            p = rec.path_obj()
            if p.exists():
                p.unlink(missing_ok=True)
                # Remove empty parent dir
                try:
                    p.parent.rmdir()
                except OSError:
                    pass
        return self._registry.remove_model(model_id)
