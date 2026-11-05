#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$scriptDir"

if command -v uv >/dev/null 2>&1; then
    uvCommand="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    uvCommand="$HOME/.local/bin/uv"
else
    printf '%s\n' 'uv was not found. Install it once, then run this launcher again.' 'https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 127
fi

export PYTHONUTF8=1
exec "$uvCommand" run --locked --no-dev anima-easy "$@"
