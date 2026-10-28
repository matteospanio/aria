#!/usr/bin/env bash
# aria-live.sh - start aria-server, wait until it is ready, run aria-repl against it, and
# stop the server when the REPL exits.
#
#   scripts/aria-live.sh -m <model_dir> [server args...] [-- repl args...]
#   scripts/aria-live.sh -m models/small-music --steer-dir directions/small -- --record take.wav
#
# env: ARIA_PORT (default 8321), ARIA_LOG (default aria-server.log),
#      ARIA_BIN (directory with aria-server and aria-repl; default: the repo root)
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
bin=${ARIA_BIN:-$here}; port=${ARIA_PORT:-8321}; log=${ARIA_LOG:-aria-server.log}
srv_args=(); repl_args=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; repl_args=("$@"); break; fi
  srv_args+=("$1"); shift
done
if [ ! -x "$bin/aria-server" ] || [ ! -x "$bin/aria-repl" ]; then
  echo "aria-live: build the binaries first (make live)" >&2; exit 1
fi
"$bin/aria-server" --port "$port" --log "$log" "${srv_args[@]}" 2>>"$log" &
srv=$!
cleanup() { kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; stty sane 2>/dev/null; }
trap cleanup EXIT
trap 'exit 130' INT TERM
# the server listens only after loading the model and probing its speed
echo "aria-live: starting aria-server on port $port (log: $log) ..." >&2
until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
  if ! kill -0 "$srv" 2>/dev/null; then
    echo "aria-live: aria-server exited during startup:" >&2; tail -5 "$log" >&2; exit 1
  fi
  sleep 0.2
done
grep -E "probe:" "$log" | tail -1 | cut -c25- >&2
"$bin/aria-repl" --host "127.0.0.1:$port" "${repl_args[@]}"
