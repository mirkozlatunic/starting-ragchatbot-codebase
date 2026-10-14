#!/bin/bash
# Auto-format all Python code with black and apply ruff's safe lint fixes.
set -e
cd "$(dirname "$0")/.."
uv run black .
uv run ruff check --fix .
