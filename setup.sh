#!/usr/bin/env bash
# One-time setup: clone the Meta dataset repo and create the conda env "face_events".
# Usage: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d seamless_interaction ]; then
  git clone https://github.com/facebookresearch/seamless_interaction.git
fi

if conda env list | grep -qE '^face_events\s'; then
  conda env update -n face_events -f environment.yml
else
  conda env create -f environment.yml
fi
# editable install: fs.py finds assets/filelist.csv relative to the clone
conda run -n face_events pip install -e ./seamless_interaction

conda run -n face_events python -c "import seamless_interaction.fs, cv2, numpy; print('python ok')"
conda run -n face_events eventcv simulate --help > /dev/null && echo "eventcv ok"
encoders=$(conda run -n face_events ffmpeg -hide_banner -encoders 2>/dev/null)
grep -q libx264 <<< "$encoders" && echo "ffmpeg+libx264 ok" || { echo "ffmpeg lacks libx264" >&2; exit 1; }
echo "Setup done. Run: conda activate face_events"
