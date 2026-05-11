# Public Beta Release Gates

OpenDrop is **beta-ready** only when all gates below are marked PASS.

## 1) Stability
- PASS when CI lint/tests/type/package are green on main.
- PASS when critical CLI smoke tests pass (`pull/search/run/serve/fine-tune/tui/hardware`).
- FAIL on any crash in standard quickstart flow.

## 2) Security
- PASS when CodeQL workflow is enabled and clean for the release commit.
- PASS when no known secrets are present in repository or docs.
- FAIL on unresolved high-severity vulnerability in changed code.

## 3) Documentation
- PASS when README quickstart commands are validated.
- PASS when troubleshooting + limitations docs are present and current.
- FAIL when onboarding requires undocumented steps.

## 4) Packaging
- PASS when `python -m build` succeeds for sdist and wheel.
- PASS when package metadata/version/changelog are consistent.
- FAIL when build artifacts cannot be produced from a clean checkout.

## 5) Compatibility
- PASS when tests pass on supported Python versions in CI.
- PASS when defaults work on CPU-only environments.
- FAIL when a supported Python version regresses.

## 6) Observability
- PASS when startup and failure paths provide actionable CLI/server errors.
- PASS when operators can identify running models and health state.
- FAIL when expected failures are silent or ambiguous.

## 7) Support Readiness
- PASS when issue templates and PR template are present.
- PASS when known risks/limitations and rollback guidance are published.
- FAIL when users lack clear channels for reporting problems.
