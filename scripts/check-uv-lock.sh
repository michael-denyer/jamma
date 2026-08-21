#!/bin/sh
# Pre-commit hook: verify uv.lock is in sync with pyproject.toml
uv lock --check || {
    echo "uv.lock is out of sync with pyproject.toml. Run: uv lock"
    exit 1
}
