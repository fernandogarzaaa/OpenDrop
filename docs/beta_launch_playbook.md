# Public Beta Launch Playbook

## Announcement draft

OpenDrop Public Beta is now available. You can pull open-weight models from Hugging Face, run them locally through an OpenAI-compatible API, and use built-in tooling for conversion, tuning, and operations.

## Risk disclosures

- Beta software; interfaces and behavior may change.
- Hardware and model-specific edge cases are expected.
- Performance and memory characteristics vary significantly by model/backend.

## Migration guidance

- Back up `~/.local/share/opendrop/registry.db` before upgrading.
- Keep model artifacts under `~/.local/share/opendrop/models`.
- Re-run `opendrop list` and `opendrop info <id>` after upgrade to validate registry integrity.

## Rollback guidance

1. Stop any running OpenDrop services.
2. Reinstall previous package version.
3. Restore backed-up registry DB if needed.
4. Validate with `opendrop list` and `opendrop hardware`.

## Support channels

- Use GitHub bug/feature issue templates for public reports.
- Use GitHub private security advisory for vulnerabilities.
