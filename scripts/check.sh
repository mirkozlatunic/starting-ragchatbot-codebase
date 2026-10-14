#!/bin/bash
# Run all quality checks without modifying files: black, ruff and pytest.
cd "$(dirname "$0")/.."
status=0
echo "==> black --check"; uv run black --check . || status=1
echo "==> ruff check"; uv run ruff check . || status=1
echo "==> pytest"; uv run pytest backend -q || status=1
exit $status
