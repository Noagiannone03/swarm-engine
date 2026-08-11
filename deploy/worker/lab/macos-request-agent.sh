#!/bin/zsh
set -euo pipefail

# Product-equivalent Request Agent launcher for the fixed Qwen laboratory
# swarm. Secrets are read inside this process and never placed in argv.
state="$HOME/.local/share/fabi"
runtime="$state/runtime"
source_dir="$runtime/parallax-src"
account_token_file="${FABI_ACCOUNT_TOKEN_FILE:-$HOME/.config/fabi/account-token}"
registry_root="${FABI_MODEL_REGISTRY_ROOT:-$state/trust/model-registry-root-322767d6181161a6a6d1457849b1780870c59abe527b0e1775ddd914e6ed5d7a.json}"
ready_file="${FABI_REQUEST_AGENT_READY_FILE:-$state/request-agent/qwen3-0-6b-v3/frontend/ready-lab.json}"

[[ -x "$runtime/parallax-venv/bin/fabi-request-agent" ]] || {
  print -u2 "Request Agent entrypoint not found"
  exit 1
}
[[ -r "$account_token_file" ]] || { print -u2 "account token file is not readable"; exit 1; }
[[ -r "$registry_root" ]] || { print -u2 "pinned model-registry root is not readable"; exit 1; }

export FABI_ACCOUNT_TOKEN="$(<"$account_token_file")"
export FABI_NETWORK_TRANSPORT=iroh
export FABI_RELAY_URL="${FABI_RELAY_URL:-https://server.undefinedstudio.fr:4443}"
export FABI_RELAY_ENROLLMENT_URL="${FABI_RELAY_ENROLLMENT_URL:-https://server.undefinedstudio.fr/fabi-registry/v1/network/enroll}"
unset FABI_RELAY_TOKEN FABI_RELAY_TOKEN_FILE
export FABI_NETWORK_IDENTITY_PATH="${FABI_NETWORK_IDENTITY_PATH:-$state/network/request-agent-qwen3-0-6b-v3.key}"
export FABI_CATALOG_DHT_MODE=client
export FABI_CATALOG_DHT_BOOTSTRAPS="${FABI_CATALOG_DHT_BOOTSTRAPS:-[
  \"/ip4/37.59.98.16/tcp/19192/p2p/12D3KooWMQrc1rWXwaeQcshtANiw9FyyGWmfqAnVsStGRqsJ54Yi\",
  \"/ip4/37.59.98.16/tcp/19193/p2p/12D3KooWG8jJaC1upci3eDZ7XSobTPC5hT6bdqFGzzptH8q7b1eG\"
]}"
export FABI_CATALOG_DHT_IDENTITY_PATH="${FABI_CATALOG_DHT_IDENTITY_PATH:-$state/network/request-agent-catalog-qwen3-0-6b-v3.key}"
export FABI_CATALOG_DHT_LISTEN_ADDRESS=/ip4/127.0.0.1/tcp/0
export FABI_MODEL_REGISTRY_ROOT="$registry_root"
export FABI_MODEL_REGISTRY_METADATA_URL="${FABI_MODEL_REGISTRY_METADATA_URL:-https://server.undefinedstudio.fr/fabi-swarm-registry-v3/root3/metadata/}"
export FABI_MODEL_REGISTRY_TARGETS_URL="${FABI_MODEL_REGISTRY_TARGETS_URL:-https://server.undefinedstudio.fr/fabi-swarm-registry-v3/root3/targets/}"
export FABI_REQUEST_AGENT_MODEL_SWARM_ID="${FABI_REQUEST_AGENT_MODEL_SWARM_ID:-18b52f3789641d5da1352d42d072ec361dd29da356841a887c4a41ad4e7d6081}"
export FABI_REQUEST_AGENT_AUTHORITY_URL="${FABI_REQUEST_AGENT_AUTHORITY_URL:-https://server.undefinedstudio.fr/fabi-scheduler/qwen3-0-6b-v3}"
export FABI_REQUEST_AGENT_STATE_DIR="${FABI_REQUEST_AGENT_STATE_DIR:-$state/request-agent/qwen3-0-6b-v3}"
export FABI_FORCE_RELAY="${FABI_FORCE_RELAY:-0}"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p "${ready_file:h}" "$FABI_REQUEST_AGENT_STATE_DIR" "${FABI_NETWORK_IDENTITY_PATH:h}"
rm -f "$ready_file"
exec "$runtime/parallax-venv/bin/fabi-request-agent" \
  --host 127.0.0.1 \
  --port "${FABI_REQUEST_AGENT_PORT:-7778}" \
  --ready-file "$ready_file"
