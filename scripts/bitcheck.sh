#!/usr/bin/env bash
# bitcheck.sh - byte-identity gate for exact optimizations: run two aria binaries on
# the same requests and cmp the WAVs.
#
#   scripts/bitcheck.sh <base_bin> <new_bin> <model_dir>
#
# env: DEV=cpu|cuda (default cpu)  SECS="10 60"  PROMPT="..." (adds a text case when
#      the model has the tokenizer)  EXTRA="--precision fp16 ..."  (args for both runs)
#      ENVS="ARIA_DEC_WINDOW=0 ..." (env for both runs)
set -u
base=$1 new=$2 model=$3
DEV=${DEV:-cpu} SECS=${SECS:-"10 60"} EXTRA=${EXTRA:-} ENVS=${ENVS:-}
PROMPT=${PROMPT:-"warm romantic piano, slow, tender"}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
inputs=("--uncond")
[ -f "$model/t5gemma-b-b-ul2/aria_tokenizer.bin" ] && inputs+=("-p")
fail=0
for d in $SECS; do
  for in in "${inputs[@]}"; do
    args=(-m "$model" -d "$d" -s 8 --seed 1 --device "$DEV" $EXTRA)
    if [ "$in" = "-p" ]; then args+=(-p "$PROMPT"); tag=prompt; else args+=(--uncond); tag=uncond; fi
    env $ENVS "$base" "${args[@]}" -o "$tmp/base.wav" >/dev/null 2>"$tmp/base.err" || { echo "ERR  base d=$d $tag"; tail -2 "$tmp/base.err"; fail=1; continue; }
    env $ENVS "$new"  "${args[@]}" -o "$tmp/new.wav"  >/dev/null 2>"$tmp/new.err"  || { echo "ERR  new  d=$d $tag"; tail -2 "$tmp/new.err"; fail=1; continue; }
    if cmp -s "$tmp/base.wav" "$tmp/new.wav"; then echo "OK   d=$d $tag dev=$DEV $EXTRA $ENVS"
    else echo "DIFF d=$d $tag dev=$DEV $EXTRA $ENVS"; fail=1; fi
  done
done
exit $fail
