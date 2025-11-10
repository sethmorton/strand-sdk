# Contributing to Strand SDK

## Prerequisites

- Python 3.11+
- `pip`, `ruff`, `pytest`
- Familiarity with the vertical slice architecture described in `GITHUB_REPO_STRUCTURE.md`

## Workflow

1. Fork/branch from `main`.
2. Keep every feature self-contained (code, docs, tests, fixtures, benchmarks).
3. Maintain type safety (use `Protocol`, `TypedDict`, or type inference from schemas; avoid `Any`).
4. Run `pytest` and `ruff check --fix` before opening a PR.
5. Update docs/examples/tests alongside code. No placeholder tests that simply assert `True`.

## Commit Style

- Follow https://cbea.ms/git-commit/
- Message structure: `<scope>: <imperative summary>`
- Reference issue IDs when available.

## Code Review Checklist

- No prop drilling; rely on shared state or dependency injection if data crosses boundaries.
- Logging: prefer concise, structured logging in `strand.utils.logging`.
- Tests validate behavior, not implementation details.
- Avoid temporary workarounds—design long-term fixes.
