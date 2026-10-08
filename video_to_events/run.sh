#!/bin/bash
# Video -> events with EventCV, using the settings validated on the face640 clip.
#
# Usage:
#   ./run.sh INPUT.mp4 OUTPUT.h5 [--preview] [extra eventcv simulate flags...]
#
# Examples:
#   ./run.sh face640.mp4 face640.h5
#   ./run.sh face640.mp4 face640.h5 --preview                  # also writes face640_preview.mp4
#   ./run.sh face640.mp4 face640.h5 --threshold 0.3 --scale 480 480
#
# Extra flags go straight to `eventcv simulate` (see `eventcv simulate --help`) and
# override the defaults below. Output extension picks the format (.h5 recommended).
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n 2,13p "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

IN="$1"; OUT="$2"; shift 2
PREVIEW=false
EXTRA=()
for a in "$@"; do
    if [[ "$a" == "--preview" ]]; then PREVIEW=true; else EXTRA+=("$a"); fi
done

ECV="${EVENTCV:-$HOME/miniconda3/envs/eventcv/bin/eventcv}"
[[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Noise off, 8x sub-steps; everything else eventcv defaults (threshold 0.2, sigma 0.03,
# refractory 100 us, cutoff 200 Hz).
"$ECV" simulate "$IN" "$OUT" \
    --shot-noise-rate-hz 0 --leak-rate-hz 0 --upsample 8 \
    "${EXTRA[@]}"

if $PREVIEW; then
    FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$IN")
    DT_MS=$(python3 -c "n, d = '$FPS'.split('/'); print(1000 * float(d) / float(n))")
    "$ECV" render "$OUT" "${OUT%.*}_preview.mp4" --dt-ms "$DT_MS" --fps 30
fi
