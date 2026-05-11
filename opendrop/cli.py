"""OpenDrop CLI — `opendrop` command entry point.

All commands use Click.  Sub-commands:
  pull          Resolve, download, quantize, and register a model.
  search        Search models on Hugging Face.
  run           Start the inference server for a specific model.
  serve         Start the multi-model OpenAI-compatible server.
  list          List all models in the registry.
  info          Show detailed info about a model.
  rm            Remove a model from the registry (+ optionally disk).
  fine-tune     Fine-tune a model with a dataset.
  convert       Convert a local SafeTensors model to GGUF.
  tui           Launch the Textual terminal dashboard.
  hardware      Show the detected hardware profile.
  config        Show the active configuration.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

def _token_option(f):
    return click.option(
        "--token", "-t", envvar="HF_TOKEN", default=None,
        help="HuggingFace access token (also read from HF_TOKEN env var).",
    )(f)


def _quant_option(f):
    return click.option(
        "--quant", "-q", default=None,
        help="Force a specific quantization (e.g. Q4_K_M). Auto-detected if omitted.",
    )(f)


def _section_values(section: object) -> dict[str, object]:
    """Return a printable key/value mapping from a config section object."""
    if is_dataclass(section) and not isinstance(section, type):
        values: dict[str, object] = asdict(section)
    else:
        values = vars(section)
    if len(values) == 1:
        only_value = next(iter(values.values()))
        if isinstance(only_value, dict):
            return only_value
    return values


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="opendrop")
def main() -> None:
    """OpenDrop — Universal open-weight local AI aggregator."""


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

@main.command()
@click.argument("source")
@_token_option
@_quant_option
@click.option("--force", is_flag=True, help="Re-download even if already registered.")
def pull(source: str, token: str | None, quant: str | None, force: bool) -> None:
    """Pull a model from SOURCE (HuggingFace URL, org/model ID, or local path)."""
    from opendrop.core.orchestrator import Orchestrator

    try:
        orch = Orchestrator()
        orch.pull(source, token=token, quant_override=quant, force=force)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@main.command()
@click.argument("query")
@_token_option
@click.option("--limit", "-n", default=10, show_default=True, type=int, help="Max results.")
def search(query: str, token: str | None, limit: int) -> None:
    """Search Hugging Face models by QUERY."""
    from opendrop.core.resolver import search_models

    try:
        results = search_models(query, limit=limit, token=token)
    except Exception as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        sys.exit(1)

    if not results:
        console.print(f"[dim]No models found for '{query}'.[/dim]")
        return

    table = Table(title=f"Search Results: {query}", show_header=True, header_style="bold cyan")
    table.add_column("Model ID", style="bold")
    table.add_column("Task")
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")
    table.add_column("License")
    table.add_column("Updated")

    for item in results:
        table.add_row(
            item.model_id,
            item.pipeline_tag or "—",
            f"{item.downloads:,}",
            f"{item.likes:,}",
            item.license_id or "—",
            item.last_modified.split("T", 1)[0] if item.last_modified else "—",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model_id")
@click.option("--port", "-p", default=None, type=int,
              help="Port to serve on (default: auto-allocated 11401+).")
@click.option("--ctx", default=None, type=int,
              help="Context size (default: from config).")
@click.option("--no-flash-attn", is_flag=True, help="Disable flash attention.")
def run(model_id: str, port: int | None, ctx: int | None, no_flash_attn: bool) -> None:
    """Start the inference server for MODEL_ID and block."""
    from opendrop.config import get_config
    from opendrop.core.registry import Registry
    from opendrop.inference.llamacpp import LlamaCppServer, find_server_binary

    cfg = get_config()
    reg = Registry(cfg.registry_db())
    rec = reg.get_model(model_id)
    if not rec:
        console.print(
            f"[red]Model '{model_id}' not found.[/red] Run `opendrop list` to see available models."
        )
        sys.exit(1)

    gguf = Path(rec.path)
    if not gguf.exists():
        console.print(f"[red]GGUF file missing:[/red] {gguf}")
        sys.exit(1)

    binary = find_server_binary()
    if not binary:
        console.print(
            "[red]llama-server not found.[/red]\n"
            "Install llama.cpp: brew install llama.cpp  or  "
            "https://github.com/ggml-org/llama.cpp#build"
        )
        sys.exit(1)

    p = port or cfg.server.port + 1
    srv = LlamaCppServer(
        gguf_path=gguf,
        port=p,
        ctx_size=ctx or cfg.inference.context_size,
        gpu_layers=cfg.inference.gpu_layers,
        parallel=cfg.inference.parallel,
        flash_attn=not no_flash_attn,
        binary=binary,
    )
    console.print(f"[bold]Starting[/bold] {rec.display_name} on port {p} …")
    try:
        srv.start()
        reg.set_port(rec.id, p)
        console.print(
            f"[green]✓ Server ready:[/green] http://127.0.0.1:{p}\n"
            f"  OpenAI base URL: http://127.0.0.1:{p}/v1\n"
            "  Press Ctrl-C to stop."
        )
        # Block until interrupted (cross-platform).
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        reg.set_port(rec.id, None)
        console.print("[dim]Server stopped.[/dim]")


# ---------------------------------------------------------------------------
# serve  (multi-model OpenAI proxy)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--host", default=None, help="Bind host (default: from config).")
@click.option("--port", "-p", default=None, type=int, help="Bind port (default: 11400).")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the multi-model OpenAI-compatible API server (with Web UI)."""
    import uvicorn

    from opendrop.config import get_config
    from opendrop.inference.server import create_app
    from opendrop.ui.web import mount_web_ui

    cfg = get_config()
    h = host or cfg.server.host
    p = port or cfg.server.port

    console.print(f"[bold]OpenDrop serving on[/bold] http://{h}:{p}")
    console.print(f"  API:    http://{h}:{p}/v1")
    console.print(f"  Web UI: http://{h}:{p}/")
    console.print("  Press Ctrl-C to stop.\n")

    app = create_app(allow_cors=cfg.server.cors)
    mount_web_ui(app)
    uvicorn.run(app, host=h, port=p, reload=reload, log_level="info")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@main.command(name="list")
def list_models() -> None:
    """List all models in the registry."""
    from opendrop.config import get_config
    from opendrop.core.registry import Registry
    from opendrop.inference.llamacpp import get_manager

    cfg = get_config()
    reg = Registry(cfg.registry_db())
    records = reg.list_models()

    if not records:
        console.print("[dim]No models in registry. Run `opendrop pull <url>` to get started.[/dim]")
        return

    running = set(get_manager().running_models().keys())
    table = Table(title="OpenDrop Models", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("Arch")
    table.add_column("Params", justify="right")
    table.add_column("Quant")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Port", justify="right")

    for rec in records:
        is_running = rec.id in running
        status = "[green]running[/green]" if is_running else "[dim]idle[/dim]"
        table.add_row(
            rec.id,
            rec.display_name,
            rec.architecture or "—",
            f"{rec.params_b:.1f}B" if rec.params_b else "—",
            rec.quant,
            rec.size_human(),
            status,
            str(rec.server_port) if rec.server_port else "—",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model_id")
def info(model_id: str) -> None:
    """Show detailed information about MODEL_ID."""
    from opendrop.config import get_config
    from opendrop.core.registry import Registry

    cfg = get_config()
    reg = Registry(cfg.registry_db())
    rec = reg.get_model(model_id)

    if not rec:
        console.print(f"[red]Model '{model_id}' not found.[/red]")
        sys.exit(1)

    table = Table(show_header=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    for k, v in {
        "ID":          rec.id,
        "Name":        rec.display_name,
        "Model ID":    rec.model_id,
        "Source":      rec.source_url,
        "Architecture": rec.architecture or "—",
        "Parameters":  f"{rec.params_b:.2f}B" if rec.params_b else "—",
        "Quantization": rec.quant,
        "Format":      rec.format,
        "Size":        rec.size_human(),
        "Path":        rec.path,
        "License":     rec.license_id or "—",
        "Pipeline":    rec.pipeline_tag or "—",
        "Tags":        ", ".join(rec.tags[:10]) if rec.tags else "—",
        "Added":       rec.added_at,
        "Last used":   rec.last_used or "never",
        "Port":        str(rec.server_port) if rec.server_port else "—",
    }.items():
        table.add_row(k, str(v))

    console.print(table)

    if rec.license_warning:
        console.print(f"\n[yellow]⚠ License note:[/yellow] {rec.license_warning}")

    adapters = Registry(cfg.registry_db()).list_adapters(rec.id)
    if adapters:
        console.print(f"\n[bold]Adapters ({len(adapters)}):[/bold]")
        for a in adapters:
            console.print(f"  • {a.name} ({a.method}) — {a.path}")


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model_id")
@click.option("--keep-files", is_flag=True, help="Remove from registry but keep model files.")
@click.confirmation_option(prompt="Remove model from registry?")
def rm(model_id: str, keep_files: bool) -> None:
    """Remove MODEL_ID from the registry (and its GGUF file by default)."""
    from opendrop.core.orchestrator import Orchestrator

    orch = Orchestrator()
    ok = orch.remove(model_id, delete_files=not keep_files)
    if ok:
        console.print(f"[green]✓ Removed:[/green] {model_id}")
    else:
        console.print(f"[red]Model '{model_id}' not found.[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# fine-tune
# ---------------------------------------------------------------------------

@main.command(name="fine-tune")
@click.argument("model_id")
@click.option("--data", "-d", required=True,
              help="Dataset file (JSONL/CSV/TXT) or HuggingFace dataset ID.")
@click.option("--method", "-m", default="lora",
              type=click.Choice(["lora", "qlora", "full", "mlx"]),
              help="Training method (default: lora).")
@click.option("--epochs", "-e", default=3, show_default=True, type=int)
@click.option("--rank", default=16, show_default=True, type=int, help="LoRA rank.")
@click.option("--lr", default=2e-4, show_default=True, type=float, help="Learning rate.")
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--output", "-o", default=None,
              help="Output directory (default: ~/.local/share/opendrop/adapters/<model>/).")
@_token_option
@click.option("--no-gguf", is_flag=True, help="Skip GGUF conversion after training.")
def fine_tune(
    model_id: str,
    data: str,
    method: str,
    epochs: int,
    rank: int,
    lr: float,
    batch_size: int,
    output: str | None,
    token: str | None,
    no_gguf: bool,
) -> None:
    """Fine-tune MODEL_ID with a dataset."""
    from opendrop.config import get_config
    from opendrop.core.registry import Registry
    from opendrop.training.finetune import TrainingConfig
    from opendrop.training.finetune import fine_tune as do_fine_tune

    cfg = get_config()
    reg = Registry(cfg.registry_db())
    rec = reg.get_model(model_id)

    # Determine base model: use HF model_id from registry if available
    base = rec.model_id if rec else model_id

    out_dir = Path(output) if output else cfg.adapters_dir() / model_id
    train_cfg = TrainingConfig(
        method=method,
        lora_rank=rank,
        learning_rate=lr,
        num_epochs=epochs,
        batch_size=batch_size,
    )

    console.print(
        f"[bold]Fine-tuning[/bold] {base}\n"
        f"  Method  : {method}\n"
        f"  Dataset : {data}\n"
        f"  Epochs  : {epochs}\n"
        f"  Output  : {out_dir}\n"
    )

    try:
        result = do_fine_tune(
            model_id=base,
            dataset_source=data,
            output_dir=out_dir,
            cfg=train_cfg,
            token=token,
            produce_gguf=not no_gguf,
        )
        console.print("\n[bold green]Training complete![/bold green]")
        if result.adapter_dir:
            console.print(f"  Adapter : {result.adapter_dir}")
        if result.merged_gguf:
            console.print(f"  GGUF    : {result.merged_gguf}")
            console.print("\nTo register the fine-tuned GGUF:")
            console.print(f"  opendrop pull {result.merged_gguf}")
    except Exception as exc:
        console.print(f"[red]Training failed:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model_path")
@_quant_option
@click.option("--output", "-o", default=None,
              help="Output directory (default: <model_path>/../gguf/).")
@click.option("--keep-fp16", is_flag=True, help="Keep the intermediate fp16 GGUF.")
def convert(model_path: str, quant: str | None, output: str | None, keep_fp16: bool) -> None:
    """Convert a local SafeTensors model to GGUF.

    MODEL_PATH should be a directory containing .safetensors files.
    """
    from opendrop.core.converter import convert_and_quantize, needs_conversion
    from opendrop.core.quantizer import QUANT_BY_NAME

    src = Path(model_path).expanduser().resolve()
    if not needs_conversion(src):
        console.print(
            "[yellow]Path does not appear to be a SafeTensors directory.[/yellow]"
        )

    out_dir = Path(output) if output else src.parent / "gguf"
    q = quant or "Q4_K_M"

    if q not in QUANT_BY_NAME:
        console.print(f"[red]Unknown quant '{q}'.[/red]")
        sys.exit(1)

    try:
        result = convert_and_quantize(src, out_dir, q, keep_fp16=keep_fp16)
        console.print(f"[green]✓ GGUF:[/green] {result}")
        console.print(f"\nTo add to registry: opendrop pull {result}")
    except Exception as exc:
        console.print(f"[red]Conversion failed:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# tui
# ---------------------------------------------------------------------------

@main.command()
def tui() -> None:
    """Launch the Textual terminal dashboard."""
    from opendrop.ui.tui import run_tui
    run_tui()


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

@main.command()
@click.option("--quant-for", default=None, metavar="PARAMS_B",
              help="Show quantization options for a model of this size (e.g. 8 for 8B).")
def hardware(quant_for: str | None) -> None:
    """Show the detected hardware profile."""
    from opendrop.core.hardware import detect_hardware
    from opendrop.core.quantizer import quant_summary

    profile = detect_hardware()
    console.print(profile.summary())

    if quant_for:
        try:
            params = float(quant_for)
        except ValueError:
            console.print(f"[red]Invalid --quant-for value:[/red] {quant_for}")
            sys.exit(1)
        console.print("\n" + quant_summary(profile, params))


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@main.command(name="config")
def show_config() -> None:
    """Show the active configuration."""
    from platformdirs import user_config_dir

    from opendrop.config import get_config

    config_path = Path(user_config_dir("opendrop")) / "config.toml"
    cfg = get_config()
    console.print(f"[bold]Config file:[/bold] {config_path}")
    console.print(f"  [dim]{'(exists)' if config_path.exists() else '(using defaults)'}[/dim]\n")

    sections = {
        "server": cfg.server,
        "storage": cfg.storage,
        "inference": cfg.inference,
        "training": cfg.training,
    }
    for section_name, section in sections.items():
        console.print(f"[bold cyan][{section_name}][/bold cyan]")
        for k, v in _section_values(section).items():
            console.print(f"  {k} = {v!r}")
        console.print()
