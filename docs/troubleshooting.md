# Troubleshooting

## `opendrop run` says `llama-server not found`

Install llama.cpp and ensure `llama-server` is on `PATH`.

- macOS: `brew install llama.cpp`
- Linux: build from source: <https://github.com/ggml-org/llama.cpp#build>

## Pull fails with Hugging Face auth errors

- Set token explicitly: `opendrop pull <source> --token <token>`
- Or export `HF_TOKEN` and retry.

## Model appears in registry but won’t run

- Check model path exists: `opendrop info <model-id>`
- If missing, remove and re-pull:
  - `opendrop rm <model-id>`
  - `opendrop pull <source>`

## API requests return 404 model not found

- Confirm exact model ID from `opendrop list`.
- Ensure you started `opendrop serve` or `opendrop run <id>`.

## TUI does not show running models

- The TUI reads current in-process manager state and registry info.
- Refresh with `r` or restart TUI after starting/stopping servers.
