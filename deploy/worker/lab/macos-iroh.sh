#!/bin/zsh
set -euo pipefail

runtime="$HOME/.local/share/fabi/runtime"
state="$HOME/.local/share/fabi"
python="$runtime/parallax-venv/bin/python"
source_pointer="$state/runtime-candidate-current.txt"
scheduler_pointer="$state/scheduler-endpoint.txt"
source_dir="${FABI_PARALLAX_SOURCE:-$(<"$source_pointer")}"
scheduler_endpoint="${FABI_SCHEDULER_ENDPOINT:-$(<"$scheduler_pointer")}"
account_token_file="${FABI_ACCOUNT_TOKEN_FILE:-$HOME/.config/fabi/account-token}"
relay_token_file="${FABI_RELAY_TOKEN_FILE:-$state/network/relay.env}"

[[ -x "$python" ]] || { print -u2 "worker Python not found: $python"; exit 1; }
[[ -f "$source_dir/pyproject.toml" ]] || { print -u2 "invalid Parallax source: $source_dir"; exit 1; }
[[ ${#scheduler_endpoint} -eq 64 && "$scheduler_endpoint" != *[^0-9a-f]* ]] || {
  print -u2 "invalid Iroh scheduler endpoint"
  exit 1
}
[[ -r "$account_token_file" ]] || { print -u2 "account token file is not readable"; exit 1; }
[[ -r "$relay_token_file" ]] || { print -u2 "relay token file is not readable"; exit 1; }

export FABI_ACCOUNT_TOKEN="$(<"$account_token_file")"
export FABI_NETWORK_TRANSPORT="iroh"
export FABI_RELAY_URL="${FABI_RELAY_URL:-https://server.undefinedstudio.fr:4443}"
export FABI_RELAY_TOKEN_FILE="$relay_token_file"
export FABI_NETWORK_IDENTITY_PATH="${FABI_NETWORK_IDENTITY_PATH:-$state/network/worker.key}"
export FABI_SWARM_V3_STATE_DIR="${FABI_SWARM_V3_STATE_DIR:-$state/swarm-v3/registry}"
export FABI_SWARM_V3_FENCE_DB="${FABI_SWARM_V3_FENCE_DB:-$state/swarm-v3/control.sqlite3}"
export FABI_FORCE_RELAY="${FABI_FORCE_RELAY:-0}"
export FABI_WORKER_SESSION_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
export PARALLAX_KEY_PATH="$HOME/.config/fabi/identity"
export PARALLAX_PROCESS_LOG_DIR="$state/process-logs"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export RUST_LOG="${RUST_LOG:-info}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

mkdir -p \
  "$PARALLAX_PROCESS_LOG_DIR" \
  "${FABI_NETWORK_IDENTITY_PATH:h}" \
  "$FABI_SWARM_V3_STATE_DIR" \
  "${FABI_SWARM_V3_FENCE_DB:h}"
cd "$source_dir"
exec "$python" -m parallax.cli join \
  -s "$scheduler_endpoint" \
  -r \
  --max-batch-size 1 \
  --max-sequence-length 65536 \
  --max-num-tokens-per-batch 65536 \
  --kv-block-size 32 \
  --tcp-port 19080 \
  --udp-port 19080 \
  --log-level DEBUG
