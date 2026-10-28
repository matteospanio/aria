#!/usr/bin/env bash
# live_smoke.sh - end-to-end check of aria-server live sessions + aria-repl (needs a model).
#
#   scripts/live_smoke.sh <model_dir> [steer_dir]     (run from the repo root after make live)
#
# Drives aria-repl from a script (no audio device): a prompt, a steer change, a new
# prompt, auto-stop at 20 s. Checks the recording, the history, the server log, and that
# a control-free session is byte-identical to `aria --stream -o -` (same session code).
set -u
model=$1 steer_dir=${2:-}
port=${ARIA_PORT:-18411}
tmp=$(mktemp -d); trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$tmp"' EXIT
fail=0
ok() { echo "OK   $*"; }
bad() { echo "FAIL $*"; fail=1; }
args=(); [ -n "$steer_dir" ] && args=(--steer-dir "$steer_dir")
./aria-server -m "$model" --port "$port" --log "$tmp/server.log" "${args[@]}" 2>/dev/null &
srv=$!
until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
  kill -0 $srv 2>/dev/null || { echo "server died"; cat "$tmp/server.log"; exit 1; }; sleep 0.3
done
steer=""; [ -n "$steer_dir" ] && steer="/steer $(ls "$steer_dir" | grep -m1 '\.atns$' | sed 's/\.atns$//') 0.3"
{ echo "warm romantic piano, slow, tender"; sleep 6; [ -n "$steer" ] && echo "$steer"; sleep 5; echo "ambient pads, slow"; sleep 60; } |
  timeout 240 ./aria-repl --host "127.0.0.1:$port" --no-play --duration 20 --record "$tmp/rec.wav" --history "$tmp/h.jsonl" 2>"$tmp/repl.txt"
tail -1 "$tmp/repl.txt"
python3 - "$tmp/rec.wav" <<'PY' && ok "recording: WAV header matches its payload, >= 20 s" || bad "recording"
import struct, sys
b = open(sys.argv[1], "rb").read(); n = struct.unpack("<I", b[40:44])[0]
sys.exit(0 if n == len(b) - 44 and n / 8 / 44100 >= 20 else 1)
PY
grep -q '"type":"open"' "$tmp/h.jsonl" && grep -q '"type":"prompt"' "$tmp/h.jsonl" && grep -q '"type":"close"' "$tmp/h.jsonl" &&
  ! grep -q '"t_audio":null' "$tmp/h.jsonl" && python3 -c "import json,sys; [json.loads(l) for l in open('$tmp/h.jsonl')]" &&
  ok "history: valid JSONL, every change became audible" || bad "history"
grep -qE "\[s1\] open" "$tmp/server.log" && grep -qE "\[s1\] chunk" "$tmp/server.log" && grep -qE "\[s1\] close" "$tmp/server.log" &&
  ok "server log: open / chunk / close lines with the session id" || bad "server log"
# byte gate: a control-free session == the CLI stream over the finalized prefix
python3 - "$port" "$tmp/s.raw" <<'PY'
import json, socket, struct, sys, time
port, out = int(sys.argv[1]), sys.argv[2]
def post(path, obj):
    b = json.dumps(obj).encode(); s = socket.create_connection(("127.0.0.1", port))
    s.sendall(b"POST %s HTTP/1.1\r\nContent-Length: %d\r\n\r\n" % (path.encode(), len(b)) + b); s.recv(4096); s.close()
s = socket.create_connection(("127.0.0.1", port))
b = json.dumps({"prompt": "warm romantic piano, slow, tender", "seed": 7, "ahead": 0}).encode()
s.sendall(b"POST /stream HTTP/1.1\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
buf = b""; sess = None; got = 0; f = open(out, "wb"); hdr = True
while True:
    k = s.recv(1 << 16)
    if not k: break
    buf += k
    if hdr:
        if b"\r\n\r\n" not in buf: continue
        buf = buf.split(b"\r\n\r\n", 1)[1]; hdr = False
    while len(buf) >= 8:
        n = struct.unpack("<I", buf[4:8])[0]
        if len(buf) < 8 + n: break
        tag, pl, buf = buf[:4], buf[8:8 + n], buf[8 + n:]
        if tag == b"HELO": sess = json.loads(pl)["session"]
        elif tag == b"AUDI":
            f.write(pl[8:]); got += (len(pl) - 8) // 8
            if got > 44100 * 13 and sess: post("/stream/control", {"session": sess, "stop": True}); sess = None
PY
./aria -m "$model" -p "warm romantic piano, slow, tender" --seed 7 --stream --chunks 6 -o - 2>/dev/null > "$tmp/cli.raw"
python3 - "$tmp/s.raw" "$tmp/cli.raw" <<'PY' && ok "byte gate: server session == aria --stream" || bad "byte gate"
import sys
a = open(sys.argv[1], "rb").read(); b = open(sys.argv[2], "rb").read()
n = min(len(a), len(b)) - 11025 * 8
sys.exit(0 if n > 44100 * 8 * 8 and a[:n] == b[:n] else 1)
PY
exit $fail
