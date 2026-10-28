#!/usr/bin/env bash
# streamcheck.sh - byte-identity gate for --stream: run two aria binaries over the
# streaming cases, in file mode (-o x.wav) and raw mode (-o -), and cmp the output.
#
#   scripts/streamcheck.sh <base_bin> <new_bin> <model_dir>
#
# env: DEV=cpu|cuda (default cpu)  CHUNKS=3  EXTRA="..." (args for every run)
# Exits non-zero on any difference. With HOLD_FILE_EQ_RAW=1 the `--hold` + prompt-change
# case compares the new file-mode WAV payload against the base raw stream instead (the
# refactor made file mode match -o -; before, file mode tiled only the last drum loop).
set -u
base=$1 new=$2 model=$3
DEV=${DEV:-cpu} CHUNKS=${CHUNKS:-3} EXTRA=${EXTRA:-}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
python3 - "$tmp/d.atns" <<'PY'
import struct, math, sys
d=[0.3*math.sin(0.1*i)+0.2*math.cos(0.03*i) for i in range(1024)]
n=math.sqrt(sum(x*x for x in d)); d=[x/n*20.0 for x in d]
open(sys.argv[1],'wb').write(b'ATNS'+struct.pack('<III',1,1,0)+struct.pack('<q',len(d))+struct.pack('<%df'%len(d),*d))
PY
STEER="residual:10:$tmp/d.atns:0.2:0-7"
fail=0
# name | extra args | stdin (prompt lines, empty = none)
cases=(
  "plain||"
  "anchor|--anchor 0.5|"
  "evolve|--evolve 2|"
  "hold|--hold|"
  "steer|--steer $STEER|"
  "ramp|--steer $STEER --steer-ramp 0:0.3:tri|"
  "prompts||ambient pads\nbright synth"
  "hold+prompts|--hold|ambient pads\nbright synth"
)
for c in "${cases[@]}"; do
  IFS='|' read -r name args input <<<"$c"
  for mode in file raw; do
    for b in base new; do
      bin=$base; [ $b = new ] && bin=$new
      out="$tmp/$b.$mode"; [ $mode = file ] && o="$out.wav" || o=-
      run=( "$bin" -m "$model" -p "warm romantic piano" --seed 3 --stream --chunks "$CHUNKS" --device "$DEV" $args $EXTRA -o "$o" )
      if [ $mode = raw ]; then printf "$input" | "${run[@]}" >"$out" 2>"$tmp/$b.err"
      else printf "$input" | "${run[@]}" >/dev/null 2>"$tmp/$b.err"; fi
      [ $? -eq 0 ] || { echo "ERR  $name $mode $b"; tail -2 "$tmp/$b.err"; fail=1; }
    done
    label=$name
    if [ $mode = file ]; then a="$tmp/base.file.wav" n="$tmp/new.file.wav"; else a="$tmp/base.raw" n="$tmp/new.raw"; fi
    if [ "$name" = "hold+prompts" ] && [ $mode = file ] && [ "${HOLD_FILE_EQ_RAW:-0}" = 1 ]; then
      tail -c +45 "$n" > "$tmp/new.pcm"   # WAV payload (44-byte header)
      a="$tmp/base.raw.hold"
      printf "$input" | "$base" -m "$model" -p "warm romantic piano" --seed 3 --stream --chunks "$CHUNKS" --device "$DEV" $args $EXTRA -o - >"$a" 2>/dev/null
      n="$tmp/new.pcm"; label="$name(file==base raw)"
    fi
    if cmp -s "$a" "$n"; then echo "OK   $label $mode dev=$DEV"; else echo "DIFF $label $mode dev=$DEV"; fail=1; fi
  done
done
exit $fail
