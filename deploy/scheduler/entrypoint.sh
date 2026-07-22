#!/usr/bin/env sh
set -eu

HTTP_PORT="${PARALLAX_HTTP_PORT:-3001}"
TCP_PORT="${PARALLAX_TCP_PORT:-18080}"
UDP_PORT="${PARALLAX_UDP_PORT:-18080}"

set -- parallax run \
  --host 0.0.0.0 \
  --port "${HTTP_PORT}" \
  --tcp-port "${TCP_PORT}" \
  --udp-port "${UDP_PORT}"

if [ -n "${PARALLAX_MODEL:-}" ]; then
  set -- "$@" -m "${PARALLAX_MODEL}"
fi

if [ -n "${PARALLAX_WORKERS:-}" ]; then
  set -- "$@" -n "${PARALLAX_WORKERS}"
fi

if [ -n "${PARALLAX_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2086
  set -- "$@" ${PARALLAX_EXTRA_ARGS}
fi

printf 'Starting Parallax scheduler:\n  %s\n' "$*"

if [ "${PARALLAX_AUTO_INIT:-1}" = "1" ] && [ -n "${PARALLAX_MODEL:-}" ]; then
  "$@" &
  child="$!"

  trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' INT TERM

  (
    init_nodes="${PARALLAX_INIT_NODES:-1}"
    attempt=0
    while [ "${attempt}" -lt 60 ]; do
      if curl -fsS "http://127.0.0.1:${HTTP_PORT}/cluster/status_json" >/dev/null 2>&1; then
        curl -fsS -X POST "http://127.0.0.1:${HTTP_PORT}/scheduler/init" \
          -H 'content-type: application/json' \
          -d "{\"model_name\":\"${PARALLAX_MODEL}\",\"init_nodes_num\":${init_nodes},\"is_local_network\":false}" \
          >/dev/null 2>&1 || true
        exit 0
      fi
      attempt=$((attempt + 1))
      sleep 1
    done
  ) &

  wait "${child}"
else
  exec "$@"
fi
