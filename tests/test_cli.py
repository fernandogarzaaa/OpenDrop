"""CLI smoke tests for critical OpenDrop command flows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from opendrop.cli import main


class TestCriticalCLIFlows:
    def test_pull_flow(self, monkeypatch):
        import opendrop.core.orchestrator as orchestrator_mod

        called: dict[str, object] = {}

        class DummyOrchestrator:
            def pull(self, source, token=None, quant_override=None, force=False):
                called.update(
                    {
                        "source": source,
                        "token": token,
                        "quant_override": quant_override,
                        "force": force,
                    }
                )

        monkeypatch.setattr(orchestrator_mod, "Orchestrator", DummyOrchestrator)
        result = CliRunner().invoke(
            main,
            ["pull", "org/model", "--token", "abc", "--quant", "Q4_K_M", "--force"],
        )
        assert result.exit_code == 0
        assert called["source"] == "org/model"
        assert called["token"] == "abc"
        assert called["quant_override"] == "Q4_K_M"
        assert called["force"] is True

    def test_search_flow(self, monkeypatch):
        import opendrop.core.resolver as resolver_mod

        monkeypatch.setattr(
            resolver_mod,
            "search_models",
            lambda *_args, **_kwargs: [
                resolver_mod.ModelSearchResult(
                    model_id="org/model", downloads=123, likes=45, pipeline_tag="text-generation"
                )
            ],
        )
        result = CliRunner().invoke(main, ["search", "llama"])
        assert result.exit_code == 0
        assert "org/model" in result.output

    def test_run_flow(self, monkeypatch, tmp_path: Path):
        import opendrop.cli as cli_mod
        import opendrop.config as config_mod
        import opendrop.core.registry as registry_mod
        import opendrop.inference.llamacpp as llamacpp_mod

        gguf = tmp_path / "model.gguf"
        gguf.write_text("x", encoding="utf-8")
        rec = SimpleNamespace(id="m1", display_name="Model One", path=str(gguf))

        class DummyRegistry:
            def __init__(self, *_args, **_kwargs):
                self.ports: list[tuple[str, int | None]] = []

            def get_model(self, _model_id):
                return rec

            def set_port(self, model_id, port):
                self.ports.append((model_id, port))

        class DummyServer:
            def __init__(self, **_kwargs):
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        cfg = SimpleNamespace(
            server=SimpleNamespace(port=11400),
            inference=SimpleNamespace(context_size=4096, gpu_layers=-1, parallel=1),
            registry_db=lambda: tmp_path / "registry.db",
        )
        monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
        monkeypatch.setattr(registry_mod, "Registry", DummyRegistry)
        monkeypatch.setattr(llamacpp_mod, "find_server_binary", lambda: "/usr/bin/llama-server")
        monkeypatch.setattr(llamacpp_mod, "LlamaCppServer", DummyServer)
        import time as _time

        def _raise_kbi(_: float) -> None:
            raise KeyboardInterrupt()

        monkeypatch.setattr(_time, "sleep", _raise_kbi)

        result = CliRunner().invoke(cli_mod.main, ["run", "m1"])
        assert result.exit_code == 0
        assert "Server ready" in result.output

    def test_serve_flow(self, monkeypatch):
        import uvicorn

        import opendrop.config as config_mod
        import opendrop.inference.server as server_mod
        import opendrop.ui.web as web_mod

        called: dict[str, object] = {}

        cfg = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=11400, cors=True))
        monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
        monkeypatch.setattr(server_mod, "create_app", lambda allow_cors: {"allow_cors": allow_cors})
        monkeypatch.setattr(
            web_mod, "mount_web_ui", lambda app: called.setdefault("mounted_app", app)
        )
        monkeypatch.setattr(
            uvicorn,
            "run",
            lambda app, host, port, reload, log_level: called.update(
                {
                    "app": app,
                    "host": host,
                    "port": port,
                    "reload": reload,
                    "log_level": log_level,
                }
            ),
        )

        result = CliRunner().invoke(main, ["serve"])
        assert result.exit_code == 0
        assert called["host"] == "127.0.0.1"
        assert called["port"] == 11400

    def test_fine_tune_flow(self, monkeypatch, tmp_path: Path):
        import opendrop.config as config_mod
        import opendrop.core.registry as registry_mod
        import opendrop.training.finetune as finetune_mod

        cfg = SimpleNamespace(
            registry_db=lambda: tmp_path / "registry.db",
            adapters_dir=lambda: tmp_path / "adapters",
        )
        monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
        monkeypatch.setattr(
            registry_mod,
            "Registry",
            lambda *_args, **_kwargs: SimpleNamespace(get_model=lambda _mid: None),
        )
        monkeypatch.setattr(
            finetune_mod,
            "fine_tune",
            lambda **_kwargs: SimpleNamespace(
                adapter_dir=tmp_path / "adapters/run-1",
                merged_gguf=tmp_path / "adapters/run-1/model.gguf",
            ),
        )

        result = CliRunner().invoke(
            main,
            ["fine-tune", "org/model", "--data", "train.jsonl", "--epochs", "1"],
        )
        assert result.exit_code == 0
        assert "Training complete" in result.output

    def test_tui_flow(self, monkeypatch):
        import opendrop.ui.tui as tui_mod

        called = {"ran": False}
        monkeypatch.setattr(tui_mod, "run_tui", lambda: called.update({"ran": True}))
        result = CliRunner().invoke(main, ["tui"])
        assert result.exit_code == 0
        assert called["ran"] is True

    def test_hardware_flow(self, monkeypatch):
        import opendrop.core.hardware as hardware_mod
        import opendrop.core.quantizer as quantizer_mod

        monkeypatch.setattr(
            hardware_mod, "detect_hardware", lambda: SimpleNamespace(summary=lambda: "HW summary")
        )
        monkeypatch.setattr(
            quantizer_mod,
            "quant_summary",
            lambda _profile, params: f"quant summary for {params}",
        )
        result = CliRunner().invoke(main, ["hardware", "--quant-for", "8"])
        assert result.exit_code == 0
        assert "HW summary" in result.output
        assert "quant summary for 8.0" in result.output


class TestListSearchFlag:
    def test_list_search_filters_by_name(self, monkeypatch, tmp_path: Path):
        from types import SimpleNamespace

        import opendrop.config as config_mod
        import opendrop.core.registry as registry_mod
        import opendrop.inference.llamacpp as llamacpp_mod

        cfg = SimpleNamespace(registry_db=lambda: tmp_path / "registry.db")
        monkeypatch.setattr(config_mod, "get_config", lambda: cfg)

        recs = [
            SimpleNamespace(
                id="llama-3-8b",
                display_name="LLaMA 3 8B",
                architecture="llama",
                params_b=8.0,
                quant="Q4_K_M",
                size_bytes=4_000_000_000,
                server_port=None,
                size_human=lambda: "4.0 GB",
            ),
            SimpleNamespace(
                id="mistral-7b",
                display_name="Mistral 7B",
                architecture="mistral",
                params_b=7.0,
                quant="Q4_K_M",
                size_bytes=3_500_000_000,
                server_port=None,
                size_human=lambda: "3.5 GB",
            ),
        ]

        monkeypatch.setattr(
            registry_mod,
            "Registry",
            lambda *_a, **_kw: SimpleNamespace(list_models=lambda: recs),
        )
        monkeypatch.setattr(
            llamacpp_mod,
            "get_manager",
            lambda: SimpleNamespace(running_models=lambda: {}),
        )

        result = CliRunner().invoke(main, ["list", "--search", "llama"])
        assert result.exit_code == 0
        assert "llama-3-8b" in result.output
        assert "mistral-7b" not in result.output

    def test_list_search_no_match_shows_empty_message(self, monkeypatch, tmp_path: Path):
        from types import SimpleNamespace

        import opendrop.config as config_mod
        import opendrop.core.registry as registry_mod
        import opendrop.inference.llamacpp as llamacpp_mod

        cfg = SimpleNamespace(registry_db=lambda: tmp_path / "registry.db")
        monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
        monkeypatch.setattr(
            registry_mod,
            "Registry",
            lambda *_a, **_kw: SimpleNamespace(list_models=lambda: []),
        )
        monkeypatch.setattr(
            llamacpp_mod,
            "get_manager",
            lambda: SimpleNamespace(running_models=lambda: {}),
        )

        result = CliRunner().invoke(main, ["list", "--search", "nonexistent"])
        assert result.exit_code == 0
        assert "No models" in result.output
