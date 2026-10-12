#!/usr/bin/env bash
# One-time setup: clone the Meta dataset repo and create the conda env "face_events".
# Safe to re-run. Usage: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d seamless_interaction ]; then
  git clone https://github.com/facebookresearch/seamless_interaction.git
fi

ENV_BIN="$(conda info --base)/envs/face_events/bin"
if [ -x "$ENV_BIN/python" ]; then
  conda env update -n face_events -f environment.yml
else
  conda env create -f environment.yml
fi

# editable install with the env's own pip: fs.py finds assets/filelist.csv relative to the clone
(cd seamless_interaction && "$ENV_BIN/python" -m pip install -e .)

"$ENV_BIN/python" -I -c "import seamless_interaction.fs, cv2, numpy; print('python ok')"
"$ENV_BIN/eventcv" simulate --help > /dev/null && echo "eventcv ok"
"$ENV_BIN/ffmpeg" -hide_banner -encoders 2>/dev/null > /tmp/face_events_encoders.txt
grep -q libx264 /tmp/face_events_encoders.txt && echo "ffmpeg+libx264 ok" || { echo "ffmpeg lacks libx264" >&2; exit 1; }
rm -f /tmp/face_events_encoders.txt
echo "Setup done. Run: conda activate face_events"
