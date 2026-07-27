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
registry_root="${FABI_MODEL_REGISTRY_ROOT:-${source_dir:h}/bootstrap-root.json}"
catalog_bootstraps="${FABI_CATALOG_DHT_BOOTSTRAPS:-[
  \"/ip4/37.59.98.16/tcp/19191/p2p/12D3KooWB1VciohMDGP6qC5m1tDRbCMfjQ14LWx12FsWyJtnWEsn\",
  \"/ip4/37.59.98.16/tcp/19192/p2p/12D3KooWMQrc1rWXwaeQcshtANiw9FyyGWmfqAnVsStGRqsJ54Yi\"
]}"

[[ -x "$python" ]] || { print -u2 "worker Python not found: $python"; exit 1; }
[[ -f "$source_dir/pyproject.toml" ]] || { print -u2 "invalid Parallax source: $source_dir"; exit 1; }
[[ ${#scheduler_endpoint} -eq 64 && "$scheduler_endpoint" != *[^0-9a-f]* ]] || {
  print -u2 "invalid Iroh scheduler endpoint"
  exit 1
}
[[ -r "$account_token_file" ]] || { print -u2 "account token file is not readable"; exit 1; }
[[ -r "$relay_token_file" ]] || { print -u2 "relay token file is not readable"; exit 1; }
[[ -r "$registry_root" ]] || { print -u2 "pinned model-registry root is not readable"; exit 1; }

export FABI_ACCOUNT_TOKEN="$(<"$account_token_file")"
export FABI_NETWORK_TRANSPORT="iroh"
export FABI_RELAY_URL="${FABI_RELAY_URL:-https://server.undefinedstudio.fr:4443}"
export FABI_RELAY_TOKEN_FILE="$relay_token_file"
export FABI_NETWORK_IDENTITY_PATH="${FABI_NETWORK_IDENTITY_PATH:-$state/network/worker.key}"
export FABI_SWARM_V3_STATE_DIR="${FABI_SWARM_V3_STATE_DIR:-$state/swarm-v3/registry}"
export FABI_SWARM_V3_FENCE_DB="${FABI_SWARM_V3_FENCE_DB:-$state/swarm-v3/control.sqlite3}"
export FABI_SWARM_V3_MODE="${FABI_SWARM_V3_MODE:-active}"
export FABI_SWARM_V3_PLACEMENT="${FABI_SWARM_V3_PLACEMENT:-autonomous}"
export FABI_MODEL_REGISTRY_ROOT="$registry_root"
export FABI_MODEL_REGISTRY_METADATA_URL="${FABI_MODEL_REGISTRY_METADATA_URL:-https://server.undefinedstudio.fr/fabi-swarm-registry-v3/metadata/}"
export FABI_MODEL_REGISTRY_TARGETS_URL="${FABI_MODEL_REGISTRY_TARGETS_URL:-https://server.undefinedstudio.fr/fabi-swarm-registry-v3/targets/}"
export FABI_CATALOG_DHT_MODE="${FABI_CATALOG_DHT_MODE:-client}"
export FABI_CATALOG_DHT_BOOTSTRAPS="$catalog_bootstraps"
export FABI_CATALOG_DHT_IDENTITY_PATH="${FABI_CATALOG_DHT_IDENTITY_PATH:-$state/network/worker-catalog.key}"
export FABI_CATALOG_DHT_LISTEN_ADDRESS="${FABI_CATALOG_DHT_LISTEN_ADDRESS:-/ip4/127.0.0.1/tcp/0}"
export FABI_FORCE_RELAY="${FABI_FORCE_RELAY:-0}"
export FABI_INITIAL_ALLOCATION_TIMEOUT_SECONDS="${FABI_INITIAL_ALLOCATION_TIMEOUT_SECONDS:-0}"
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
  "${FABI_CATALOG_DHT_IDENTITY_PATH:h}" \
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
