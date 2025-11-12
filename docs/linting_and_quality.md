# Code Quality & Linting

This document explains how to run linting, type checking, and tests for the Strand SDK.

## Prerequisites

Ensure you have the development environment set up:

```bash
./scripts/setup_dev.sh
```

This will:
1. Create a Python virtual environment (`.venv`)
2. Install all dependencies (including dev tools)
3. Install the package in editable mode
4. Set up pre-commit hooks

## Quick Start

Activate the virtual environment before running any commands:

```bash
source .venv/bin/activate
```

## Linting with Ruff

Ruff is a fast Python linter that checks for style issues, unused imports, and common mistakes.

### Check for linting errors

```bash
ruff check strand/
```

To check specific files or directories:

```bash
ruff check strand/engine/
ruff check strand/rewards/base.py
```

### Auto-fix fixable errors

Ruff can automatically fix many issues:

```bash
ruff check strand/ --fix
```

To see what would be fixed without applying changes:

```bash
ruff check strand/ --fix --diff
```

## Type Checking with MyPy

MyPy performs static type analysis to catch type errors before runtime.

### Check types

```bash
mypy strand/
```

To check specific files:

```bash
mypy strand/engine/engine.py
```

## Running Tests

Use pytest to run the test suite:

```bash
pytest tests/
```

For verbose output:

```bash
pytest tests/ -v
```

With coverage reporting:

```bash
pytest tests/ --cov=strand --cov-report=html
```

## Complete Pre-PR Checklist

Before opening a pull request, run these commands:

```bash
# Activate virtual environment
source .venv/bin/activate

# Auto-fix linting issues
ruff check strand/ tests/ --fix

# Run linter to verify
ruff check strand/ tests/

# Check types
mypy strand/

# Run tests
pytest tests/ -v
```

## Pre-commit Hooks

The setup script installs pre-commit hooks that automatically run linting before each commit.

To manually run pre-commit checks:

```bash
pre-commit run --all-files
```

To skip pre-commit checks (not recommended):

```bash
git commit --no-verify
```

## Configuration

Ruff and MyPy configurations are defined in `pyproject.toml`:

```toml
[tool.ruff]
# Ruff configuration

[tool.mypy]
# MyPy configuration
```

See `pyproject.toml` for the current configuration.

## Common Issues

### "command not found: ruff"

Ensure the virtual environment is activated:

```bash
source .venv/bin/activate
```

### ImportError when running linter

Make sure the package is installed in editable mode:

```bash
pip install -e .
```

### Type errors that seem incorrect

Try running with `--strict` mode:

```bash
mypy strand/ --strict
```

This enables stricter type checking rules.

