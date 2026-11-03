#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

find . -type f \
  ! -path './results/*' \
  ! -path './.mplconfig/*' \
  ! -path './__pycache__/*' \
  ! -name 'MODEL_STRUCTURE_AUDIT.json' \
  ! -name 'FILE_MANIFEST_SHA256.txt' \
  -print0 | sort -z | xargs -0 shasum -a 256 > FILE_MANIFEST_SHA256.txt

echo "Immutable package manifest refreshed: FILE_MANIFEST_SHA256.txt"
