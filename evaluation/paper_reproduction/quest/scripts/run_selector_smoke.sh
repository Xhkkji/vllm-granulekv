#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m evaluation.paper_reproduction.quest.scripts.selector_smoke "$@"
