# OpenDrop

> Universal open-weight local AI aggregator — drop a model link, run it locally.

[![Beta](https://img.shields.io/badge/status-beta-orange)](https://github.com/fernandogarzaaa/OpenDrop)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/fernandogarzaaa/OpenDrop/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/OpenDrop/actions/workflows/ci.yml)

---

## What is OpenDrop?

OpenDrop is a single piece of software where you paste any Hugging Face (or compatible) URL pointing to an open-weight model and get a **fully running, hardware-optimized, locally-served AI** — with optional post-training via a dropped dataset — no ML knowledge required.

Inspired by the end-to-end philosophy of [ds4.c](https://github.com/fernandogarzaaa/ds4) — one model, one chip, validated deeply — OpenDrop generalizes that: *any open model, every hardware target, zero configuration friction*.

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"          # development
pip install opendrop             # production (PyPI, coming soon)

# Pull a model from Hugging Face
opendrop pull https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

# Search models on Hugging Face
opendrop search "llama 3.1 instruct"

# Run it (starts an OpenAI-compatible server on :11400)
opendrop run llama-3.1-8b-instruct

# Chat via any OpenAI client, e.g.:
curl http://localhost:11400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instruct","messages":[{"role":"user","content":"Hello!"}]}'

# Fine-tune with a local dataset
opendrop fine-tune llama-3.1-8b-instruct --data my_data.jsonl

# Terminal dashboard
opendrop tui

# What hardware do I have?
opendrop hardware
```

---

## Features

| Feature | Status |
|---|---|
| HuggingFace URL model pull | ✅ |
| Direct GGUF URL download | ✅ |
| Hardware auto-detection (Apple Silicon, NVIDIA, AMD, CPU) | ✅ |
| Automatic quantization selection | ✅ |
| SafeTensors → GGUF conversion | ✅ |
| OpenAI-compatible server (`/v1/chat/completions`) | ✅ |
| Streaming responses | ✅ |
| LoRA fine-tuning | ✅ |
| QLoRA fine-tuning (low VRAM) | ✅ |
| MLX fine-tuning (Apple Silicon) | ✅ |
| Dataset format auto-detection | ✅ |
| SQLite model registry | ✅ |
| Textual TUI dashboard | ✅ |
| Embedded Web UI | ✅ |
| Multi-model concurrent serving | ✅ |
| Disk KV cache (long context) | ✅ |
| License detection | ✅ |
| Model re-quantization | ✅ |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Interfaces                   │
│         CLI        TUI        Web UI        API      │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                   OpenDrop Core                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Model        │  │ Hardware     │  │   Model   │  │
│  │ Resolver     │  │ Profiler     │  │ Registry  │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                 │                │         │
│  ┌──────▼─────────────────▼────────────────▼──────┐  │
│  │              Orchestration Engine               │  │
│  │  (quant decision · download · conversion)      │  │
│  └───────────────────────┬─────────────────────────┘  │
└───────────────────────── ┼ ──────────────────────────┘
                           │
        ┌──────────────────┼─────────────────┐
        ▼                  ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Inference   │  │  Fine-Tune   │  │  Conversion  │
│  Backends    │  │  Pipeline    │  │  Pipeline    │
│  llama.cpp   │  │  LoRA/QLoRA  │  │  HF → GGUF   │
│  ds4.c (fast)│  │  MLX-LM      │  │  llama.cpp   │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Supported Model Sources

| Source | Example |
|---|---|
| HuggingFace model page | `https://huggingface.co/org/model` |
| HuggingFace GGUF direct file | `https://huggingface.co/org/model/resolve/main/model.Q4_K_M.gguf` |
| Local GGUF file | `/path/to/model.gguf` |
| Local SafeTensors directory | `/path/to/model/` |
| HuggingFace model ID | `org/model` |

---

## Hardware Support

| Hardware | Backend | Quantization range |
|---|---|---|
| Apple Silicon (M-series) | Metal via llama.cpp | Q2_K – fp16 |
| NVIDIA GPU (CUDA) | CUDA via llama.cpp | Q2_K – fp16 |
| AMD GPU (ROCm) | ROCm via llama.cpp | Q2_K – Q8_0 |
| Intel Arc (Vulkan) | Vulkan via llama.cpp | Q4_K_M – Q8_0 |
| CPU only | llama.cpp CPU | Q4_K_M – Q8_0 |

---

## Fine-Tuning

```bash
# LoRA (GPU/CUDA)
opendrop fine-tune my-model --data train.jsonl --method lora --epochs 3

# QLoRA (low VRAM, 8 GB+)
opendrop fine-tune my-model --data train.jsonl --method qlora --epochs 3

# Apple Silicon (MLX)
opendrop fine-tune my-model --data train.jsonl --method mlx --epochs 3

# Full fine-tune (small models, large VRAM)
opendrop fine-tune my-model --data train.jsonl --method full
```

**Supported dataset formats:**
- JSONL (instruction/response, prompt/completion, conversations)
- CSV
- HuggingFace dataset ID (`org/dataset`)
- Alpaca format
- ShareGPT format
- Raw text (for continued pre-training)

---

## CLI Reference

```
opendrop pull <url>              Pull and prepare a model
opendrop search <query>          Search models on Hugging Face
opendrop run <model-id>         Start inference server for a model
opendrop serve                  Serve all registered models
opendrop list                   List all models in registry
opendrop info <model-id>        Show model details
opendrop rm <model-id>          Remove a model from registry + disk
opendrop fine-tune <id>         Fine-tune with a dataset
opendrop convert <path>         Convert local SafeTensors → GGUF
opendrop tui                    Launch terminal dashboard
opendrop hardware               Show hardware profile
opendrop config                 Show/edit configuration
```

---

## Configuration

Config file: `~/.config/opendrop/config.toml` (Linux/macOS) or `%APPDATA%\opendrop\config.toml` (Windows).

```toml
[server]
host = "127.0.0.1"
port = 11400
cors = true

[storage]
models_dir = "~/.local/share/opendrop/models"
registry_db = "~/.local/share/opendrop/registry.db"

[inference]
context_size = 8192
disk_kv_cache = true           # Long-context disk-backed KV state
gpu_layers = -1                # -1 = auto (all layers to GPU)

[training]
default_method = "lora"
lora_rank = 16
lora_alpha = 32
learning_rate = 2e-4
batch_size = 4
gradient_accumulation = 4
```

---

## Design Principles

1. **End-to-end quality** — not just "runnable", but validated per architecture
2. **Disk KV cache as first-class** — SSDs are fast enough for long context
3. **Quantization transparency** — always show what was chosen and why
4. **Zero cloud calls during inference** — fully offline after download
5. **Fail loud** — explain hardware bottlenecks clearly

---

## Roadmap

### Beta (current)
- Core pull/run/fine-tune/list/rm CLI
- llama.cpp Metal + CUDA backends
- LoRA + QLoRA training
- Textual TUI + embedded Web UI

### v1.0
- Vision model support
- Embedding model server
- Model re-quantization UI
- Windows DirectML backend

### v1.1
- Distributed multi-node inference
- Model sharing / export
- Community model driver plugins

---

## Acknowledgements

OpenDrop is built on the shoulders of:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the inference engine that makes local AI possible
- **[ds4.c](https://github.com/fernandogarzaaa/ds4)** — the inspiration for end-to-end local model quality
- **[HuggingFace Hub](https://huggingface.co/)** — the open-model ecosystem
- **[Textual](https://github.com/Textualize/textual)** — the TUI framework

---

## License

MIT — see [LICENSE](LICENSE)

---

## Public Beta Readiness Docs

- [Beta release gates](docs/beta_release_gates.md)
- [Beta readiness report](docs/beta_readiness_report.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Known limitations](docs/known_limitations.md)
- [Beta launch playbook](docs/beta_launch_playbook.md)
- [Changelog](CHANGELOG.md)
