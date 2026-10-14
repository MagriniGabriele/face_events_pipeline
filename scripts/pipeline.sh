#!/usr/bin/env bash
# Resumable pipeline: for each archive in assets/filelist.csv, download -> face crop -> cleanup.
# Usage: setsid nohup scripts/pipeline.sh [--label L] [--split S] [--batches 0-99] [--limit N] > /dev/null 2>&1 &
#   --batches:   dataset batch indices, e.g. "3", "0-99", "0,5,10-12" (default: all)
#   --limit:     max number of ARCHIVES (~1 GB, 4-19 clips each) to process in this run
#   --max-clips: max clips per archive (smoke test); such an archive is left unfinished
# Env: SI_REPO, FEP_PYTHON, MAX_CONSEC_FAIL (default 3), RAW_DIR, OUT_DIR, CROP_SIZE, EVENTS=1 (simulate events), EVENTS_DEVICE=gpu|cpu, EVENTS_SCALE="W H", EVENTS_MAX_FRAMES, MIN_FREE_GB
set -u
cd "$(dirname "$0")/.."
SI_REPO="${SI_REPO:-seamless_interaction}"  # clone of facebookresearch/seamless_interaction
# interpreter: FEP_PYTHON if set, else the active env's python (a generic $PYTHON from the shell is ignored)
PYTHON="${FEP_PYTHON:-$(command -v python || command -v python3)}"

RAW_DIR="${RAW_DIR:-$HOME/datasets/seamless_interaction_raw}"
OUT_DIR="${OUT_DIR:-$HOME/datasets/seamless_interaction_faces}"
CROP_SIZE="${CROP_SIZE:-640}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
LABEL=""; SPLIT=""; BATCHES=""; LIMIT=0; MAX_CLIPS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --batches) BATCHES="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --max-clips) MAX_CLIPS="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
mkdir -p "$RAW_DIR" "$OUT_DIR"
LOG="$OUT_DIR/pipeline.log"; FAILED="$OUT_DIR/failed.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }  # terminal + log file
die() { log "ERROR: $*"; exit 1; }
MAX_CONSEC_FAIL="${MAX_CONSEC_FAIL:-3}"

# preflight: fail loudly instead of failing every archive
[ -f "$SI_REPO/assets/filelist.csv" ] || die "$SI_REPO/assets/filelist.csv not found (run setup.sh, or set SI_REPO)"
"$PYTHON" -I -c "import seamless_interaction.fs" 2>/dev/null \
  || die "'$PYTHON' cannot import seamless_interaction: $("$PYTHON" -I -c 'import seamless_interaction.fs' 2>&1 | tail -1)"
log "python: $PYTHON ($("$PYTHON" -c 'import numpy, sys; print(sys.version.split()[0], "numpy", numpy.__version__)'))"
command -v ffmpeg > /dev/null || die "ffmpeg not found on PATH"
if [ -n "${EVENTS:-}" ]; then
  "${EVENTCV:-eventcv}" simulate --help > /dev/null 2>&1 || die "eventcv not found (EVENTS=1); set EVENTCV or activate the env"
fi

# unique archives: label split batch archive
list=$("$PYTHON" -I - "$SI_REPO/assets/filelist.csv" "$LABEL" "$SPLIT" "$BATCHES" <<'PY'
import csv, sys
filelist, label, split, batches = sys.argv[1:5]
keep = set()
for part in filter(None, batches.split(",")):
    lo, _, hi = part.partition("-")
    keep.update(range(int(lo), int(hi or lo) + 1))
s = {(r["label"], r["split"], int(r["batch_idx"]), int(r["archive_idx"]))
     for r in csv.DictReader(open(filelist))
     if (not label or r["label"] == label) and (not split or r["split"] == split)
     and (not keep or int(r["batch_idx"]) in keep)}
for a in sorted(s): print(*a)
PY
)
[ -n "$list" ] || die "no archives match label=${LABEL:-all} split=${SPLIT:-all} batches=${BATCHES:-all}"
log "start: $(echo "$list" | wc -l) archives (label=${LABEL:-all} split=${SPLIT:-all} batches=${BATCHES:-all} limit=$LIMIT max_clips=$MAX_CLIPS)"
[ "$MAX_CLIPS" -gt 0 ] && log "note: --max-clips $MAX_CLIPS leaves every archive unfinished (no _DONE), so a rerun redoes it"

n=0; consec=0
while read -r label split batch archive; do
  [ -z "$label" ] && continue
  name=$(printf '%s/%s/%04d/%04d' "$label" "$split" "$batch" "$archive")
  [ -e "$OUT_DIR/$name/_DONE" ] && continue
  if [ "$LIMIT" -gt 0 ] && [ "$n" -ge "$LIMIT" ]; then break; fi
  n=$((n + 1))  # (stdin of the worker is /dev/null: ffmpeg would otherwise eat this list)
  free=$(df -BG --output=avail "$RAW_DIR" | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${free}G free (< ${MIN_FREE_GB}G) before $name"; exit 1
  fi
  log "begin $name"; t0=$SECONDS
  extra=()
  [ -n "${EVENTS:-}" ] && extra+=(--events)
  [ "$MAX_CLIPS" -gt 0 ] && extra+=(--max-clips "$MAX_CLIPS")
  # -u: unbuffered, else the worker's per-clip progress sits in its stdout buffer
  # (invisible in $LOG) until the whole archive is done, which looks like a hang
  if "$PYTHON" -u scripts/process_archive.py "$label" "$split" "$batch" "$archive" \
      --raw "$RAW_DIR" --out "$OUT_DIR" --size "$CROP_SIZE" ${extra[@]+"${extra[@]}"} < /dev/null >> "$LOG" 2>&1; then
    [ "$MAX_CLIPS" -gt 0 ] && suffix=", partial" || suffix=""
    log "ok $name ($((SECONDS - t0))s$suffix)"; consec=0
  else
    log "FAILED $name ($((SECONDS - t0))s), last log lines:"
    grep -v ' - INFO - ' "$LOG" | tail -6 | grep -v "begin $name" | sed 's/^/    /'
    echo "$name" >> "$FAILED"; consec=$((consec + 1))
    [ "$consec" -ge "$MAX_CONSEC_FAIL" ] && die "$consec archives failed in a row, stopping (see $LOG)"
  fi
done <<< "$list"
log "finished ($n processed)"
