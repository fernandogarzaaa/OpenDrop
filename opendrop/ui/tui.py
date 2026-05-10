"""Textual TUI dashboard for OpenDrop.

Shows:
  - Model registry list with status (running / idle)
  - Hardware profile panel
  - Active server sessions with tokens/sec
  - Scrollable log panel

Launch: opendrop tui
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    RichLog,
    Static,
)

from opendrop.config import get_config
from opendrop.core.hardware import HardwareProfile, detect_hardware
from opendrop.core.registry import ModelRecord, Registry
from opendrop.inference.llamacpp import get_manager

# ---------------------------------------------------------------------------
# Hardware panel
# ---------------------------------------------------------------------------

class HardwarePanel(Static):
    """Displays the hardware profile summary."""

    DEFAULT_CSS = """
    HardwarePanel {
        border: round $accent;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, profile: HardwareProfile) -> None:
        super().__init__()
        self._profile = profile

    def on_mount(self) -> None:
        self.update(
            Text.from_markup(
                f"[bold cyan]Hardware[/bold cyan]\n{self._profile.summary()}"
            )
        )


# ---------------------------------------------------------------------------
# Model table
# ---------------------------------------------------------------------------

class ModelTable(DataTable):
    """Interactive table of registered models."""

    COLUMNS: ClassVar[list[str]] = [
        "ID", "Name", "Arch", "Params", "Quant", "Size", "Status",
    ]

    def __init__(self, records: list[ModelRecord], running_ids: set[str]) -> None:
        super().__init__()
        self._records = records
        self._running = running_ids

    def on_mount(self) -> None:
        self.add_columns(*self.COLUMNS)
        for rec in self._records:
            status = (
                Text("● running", style="bold green")
                if rec.id in self._running
                else Text("○ idle", style="dim")
            )
            self.add_row(
                rec.id,
                rec.display_name,
                rec.architecture or "—",
                f"{rec.params_b:.1f}B" if rec.params_b else "—",
                rec.quant,
                rec.size_human(),
                status,
            )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class OpenDropTUI(App):
    """OpenDrop terminal dashboard."""

    TITLE = "OpenDrop"
    SUB_TITLE = "Universal Local AI Aggregator"

    CSS = """
    Screen {
        layers: base overlay;
    }
    #main {
        height: 1fr;
    }
    #left {
        width: 65%;
        padding: 0 1;
    }
    #right {
        width: 35%;
        padding: 0 1;
    }
    ModelTable {
        height: 1fr;
        border: round $primary;
    }
    #log-panel {
        height: 20;
        border: round $secondary;
        margin-top: 1;
    }
    Label.section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config()
        self._registry = Registry(cfg.registry_db())
        self._profile = detect_hardware()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Label("Models", classes="section-title")
                records = self._registry.list_models()
                running = set(get_manager().running_models().keys())
                yield ModelTable(records, running)
                yield RichLog(id="log-panel", highlight=True, markup=True)
        with Vertical(id="right"):
            yield HardwarePanel(self._profile)
            yield self._build_server_panel()
        yield Footer()

    def _build_server_panel(self) -> Static:
        running = get_manager().running_models()
        if not running:
            content = Text("No models running.\nUse [bold]opendrop run <id>[/bold] to start one.")
        else:
            lines = [Text("Active servers:", style="bold cyan")]
            for rec_id, srv in running.items():
                lines.append(
                    Text(f"  {rec_id}  →  {srv.base_url}", style="green")
                )
            content = Text("\n").join(lines)
        panel = Static(content)
        panel.styles.border = ("round", "green")
        panel.styles.padding = (1, 1)
        panel.styles.height = "auto"
        return panel

    def action_refresh(self) -> None:
        self.exit()
        OpenDropTUI().run()

    def on_mount(self) -> None:
        log: RichLog = self.query_one("#log-panel")  # type: ignore[assignment]
        log.write("[bold]OpenDrop TUI ready.[/bold]")
        log.write(f"Registry: {get_config().registry_db()}")
        log.write(f"Models dir: {get_config().models_dir()}")


def run_tui() -> None:
    """Launch the OpenDrop TUI."""
    OpenDropTUI().run()
