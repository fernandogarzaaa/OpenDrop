# Beta Readiness Report

Date: 2026-05-11

## Baseline quality snapshot

- Ruff lint: PASS (`python -m ruff check opendrop tests`)
- Ruff format check: FAIL at baseline (repo required formatting updates)
- Tests: PASS (`97 passed`)
- Type check: FAIL at baseline (resolver/tui/cli typing issues)
- Package build: PASS (`python -m build`)

## Blockers identified

1. Formatting drift from CI policy.
2. Type-check failures in CLI/TUI/resolver.
3. Missing explicit beta release gates.
4. Missing support artifacts (issue/PR templates and launch playbook).
5. Missing CI gates for type-check/package/security.

## Remediation delivered

- Added explicit beta release gate document.
- Fixed typed reliability issues and added config input validation.
- Added CLI smoke tests for key user flows.
- Added changelog and public beta launch/troubleshooting docs.
- Added CI jobs for mypy + package build.
- Added CodeQL workflow for security scanning.
- Added issue templates and PR template.

## Remaining release action

- Ensure all CI checks are green on the release candidate commit.
