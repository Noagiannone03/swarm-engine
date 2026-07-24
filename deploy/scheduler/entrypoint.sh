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
# ``parallax run -m/-n`` initializes the scheduler in-process.  Posting a
# second /scheduler/init after startup used to tear down that fresh instance,
# reset its in-memory epochs and make every worker reload once more.
exec "$@"
