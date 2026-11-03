#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"
shasum -a 256 -c FILE_MANIFEST_SHA256.txt
echo "Immutable package files passed SHA-256 verification."
