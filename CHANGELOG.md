# Changelog

All notable changes to OpenDrop will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning with beta prereleases.

## [Unreleased]

## [0.1.0b2] - 2026-05-14

### Added
- `/v1/hardware` API endpoint exposing hardware profile as JSON.
- Web UI pull model via SSE streaming — shows real-time progress without leaving the browser.
- Web UI hardware badge now shows real GPU/memory info from the API.
- `opendrop list --search` flag for filtering local registry by name/ID/architecture.
- `Registry.search_models()` for programmatic local model search.
- TUI auto-refresh every 5 seconds — model table updates without manual key press.
- Platform-specific `llama-server` install instructions in error messages (Windows, macOS, Linux).
- Windows executable names (`llama-server.exe`) in binary discovery.
- PyPI trusted-publishing workflow (`.github/workflows/publish.yml`).
- Cross-platform signal handling in `opendrop run` (Windows-safe blocking wait).

### Fixed
- `opendrop run` no longer raises `AttributeError` on Windows due to missing `SIGTERM`.

### Changed
- Explicit beta release gates and readiness artifacts. (carried forward from b1)
- CLI smoke tests for critical user flows. (carried forward from b1)
- Config validation for safer startup failures. (carried forward from b1)

## [0.1.0b1] - 2026-05-11

### Added
- Initial beta preview release.
