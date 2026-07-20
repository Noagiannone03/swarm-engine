#!/bin/bash
# Refuse les dépendances macOS propres à la machine de build. Une release Fabi
# peut dépendre des frameworks système et de /usr/lib, mais pas de Homebrew ni
# d'un rpath non fourni par l'archive.

set -euo pipefail

binary="${1:-}"
if [[ -z "$binary" || ! -f "$binary" ]]; then
    echo "Usage: $0 /path/to/vllm-rs" >&2
    exit 2
fi

platform="${FABI_PORTABILITY_PLATFORM:-$(uname -s)}"
if [[ "$platform" != "Darwin" ]]; then
    exit 0
fi

if ! command -v otool >/dev/null 2>&1; then
    echo "otool is required to audit macOS vllm-rs dependencies" >&2
    exit 1
fi

invalid="$(
    otool -L "$binary" | awk '
        NR > 1 {
            dependency = $1
            if (dependency !~ "^/System/Library/" && dependency !~ "^/usr/lib/") {
                print dependency
            }
        }
    '
)"

if [[ -n "$invalid" ]]; then
    echo "vllm-rs contains non-portable macOS dependencies:" >&2
    printf '  %s\n' "$invalid" >&2
    exit 1
fi

echo "vllm-rs macOS dependency audit passed."
