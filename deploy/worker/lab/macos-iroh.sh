#!/bin/zsh
set -euo pipefail

runtime="$HOME/.local/share/fabi/runtime"
state="$HOME/.local/share/fabi"
manifest_path="$state/MANIFEST"
python="$runtime/parallax-venv/bin/python"
source_pointer="$state/runtime-candidate-current.txt"
scheduler_pointer="$state/scheduler-endpoint.txt"
if [[ -n "${FABI_PARALLAX_SOURCE:-}" ]]; then
  source_dir="$FABI_PARALLAX_SOURCE"
elif [[ -r "$source_pointer" ]]; then
  source_dir="$(<"$source_pointer")"
else
  source_dir="$runtime/parallax-src"
fi
if [[ -n "${FABI_SCHEDULER_ENDPOINT:-}" ]]; then
  scheduler_endpoint="$FABI_SCHEDULER_ENDPOINT"
else
  scheduler_endpoint="$(<"$scheduler_pointer")"
fi
account_token_file="${FABI_ACCOUNT_TOKEN_FILE:-$HOME/.config/fabi/account-token}"
registry_root="${FABI_MODEL_REGISTRY_ROOT:-$state/trust/model-registry-root-322767d6181161a6a6d1457849b1780870c59abe527b0e1775ddd914e6ed5d7a.json}"
catalog_bootstraps="${FABI_CATALOG_DHT_BOOTSTRAPS:-[
  \"/ip4/37.59.98.16/tcp/19192/p2p/12D3KooWMQrc1rWXwaeQcshtANiw9FyyGWmfqAnVsStGRqsJ54Yi\",
  \"/ip4/37.59.98.16/tcp/19193/p2p/12D3KooWG8jJaC1upci3eDZ7XSobTPC5hT6bdqFGzzptH8q7b1eG\"
]}"

manifest_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$manifest_path"
}

[[ -x "$python" ]] || { print -u2 "worker Python not found: $python"; exit 1; }
[[ -f "$source_dir/pyproject.toml" ]] || { print -u2 "invalid Parallax source: $source_dir"; exit 1; }
[[ -r "$manifest_path" ]] || { print -u2 "runtime manifest is not readable: $manifest_path"; exit 1; }
[[ ${#scheduler_endpoint} -eq 64 && "$scheduler_endpoint" != *[^0-9a-f]* ]] || {
  print -u2 "invalid Iroh scheduler endpoint"
  exit 1
}
[[ -r "$account_token_file" ]] || { print -u2 "account token file is not readable"; exit 1; }
[[ -r "$registry_root" ]] || { print -u2 "pinned model-registry root is not readable"; exit 1; }
execution_engine="$(manifest_value execution_engine)"
execution_device="$(manifest_value execution_device)"
[[ "$execution_engine" == "skippy" ]] || {
  print -u2 "unsupported installed execution engine: $execution_engine"
  exit 1
}
[[ "$execution_device" == "metal" ]] || {
  print -u2 "unsupported macOS Skippy execution device: $execution_device"
  exit 1
}

export FABI_ACCOUNT_TOKEN="$(<"$account_token_file")"
export FABI_NETWORK_TRANSPORT="iroh"
export FABI_RELAY_URL="${FABI_RELAY_URL:-https://server.undefinedstudio.fr:4443}"
export FABI_RELAY_ENROLLMENT_URL="${FABI_RELAY_ENROLLMENT_URL:-https://server.undefinedstudio.fr/fabi-registry/v1/network/enroll}"
unset FABI_RELAY_TOKEN FABI_RELAY_TOKEN_FILE
export FABI_NETWORK_IDENTITY_PATH="${FABI_NETWORK_IDENTITY_PATH:-$state/network/worker.key}"
export FABI_SWARM_V3_STATE_DIR="${FABI_SWARM_V3_STATE_DIR:-$state/swarm-v3/qwen3-0-6b-v3/registry}"
export FABI_SWARM_V3_FENCE_DB="${FABI_SWARM_V3_FENCE_DB:-$state/swarm-v3/qwen3-0-6b-v3/fencing.sqlite3}"
export FABI_SWARM_V3_MODE="${FABI_SWARM_V3_MODE:-active}"
export FABI_SWARM_V3_PLACEMENT="${FABI_SWARM_V3_PLACEMENT:-autonomous}"
export FABI_SWARM_V3_COORDINATION_MODE="client"
export FABI_MODEL_REGISTRY_ROOT="$registry_root"
export FABI_MODEL_REGISTRY_METADATA_URL="${FABI_MODEL_REGISTRY_METADATA_URL:-https://server.undefinedstudio.fr/fabi-swarm-registry-v3/root3/metadata/}"
export FABI_MODEL_REGISTRY_TARGETS_URL="${FABI_MODEL_REGISTRY_TARGETS_URL:-https://server.undefinedstudio.fr/fabi-swarm-registry-v3/root3/targets/}"
export FABI_CATALOG_DHT_MODE="${FABI_CATALOG_DHT_MODE:-client}"
export FABI_CATALOG_DHT_BOOTSTRAPS="$catalog_bootstraps"
export FABI_CATALOG_DHT_IDENTITY_PATH="${FABI_CATALOG_DHT_IDENTITY_PATH:-$state/network/worker-catalog.key}"
export FABI_CATALOG_DHT_LISTEN_ADDRESS="${FABI_CATALOG_DHT_LISTEN_ADDRESS:-/ip4/127.0.0.1/tcp/0}"
export FABI_FORCE_RELAY="${FABI_FORCE_RELAY:-0}"
export FABI_INITIAL_ALLOCATION_TIMEOUT_SECONDS="${FABI_INITIAL_ALLOCATION_TIMEOUT_SECONDS:-0}"
export FABI_WORKER_SESSION_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
export PARALLAX_KEY_PATH="$state/identity"
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
  --max-sequence-length 32768 \
  --max-num-tokens-per-batch 4096 \
  --kv-block-size 32 \
  --gpu-backend skippy \
  --execution-device "$execution_device" \
  --tcp-port 19080 \
  --udp-port 19080 \
  --log-level DEBUG
