#!/usr/bin/env bash
# One-time setup: clone the Meta dataset repo, create the conda env "face_events" and
# download the MediaPipe face landmark model into landmark_models/.
# Safe to re-run. Usage: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# Test for the project file, not just the directory: a clone of this repo made while
# seamless_interaction/ was committed as a bare gitlink creates it as an EMPTY directory,
# and "-d" would then skip the clone and leave pip with nothing to install.
if [ ! -f seamless_interaction/pyproject.toml ]; then
  if [ -d seamless_interaction ] && [ -n "$(ls -A seamless_interaction)" ]; then
    echo "seamless_interaction/ exists but has no pyproject.toml." >&2
    echo "Move or delete it, then re-run: rm -rf seamless_interaction && bash setup.sh" >&2
    exit 1
  fi
  rmdir seamless_interaction 2>/dev/null || true
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

# MediaPipe Face Landmarker model for scripts/generate_landmarks.py (~3.7 MB)
LANDMARK_MODEL_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
mkdir -p landmark_models
if [ ! -s landmark_models/face_landmarker.task ]; then
  # to a temp name first: an interrupted download must not look like a model
  curl -fL --retry 3 -o landmark_models/face_landmarker.task.part "$LANDMARK_MODEL_URL"
  mv landmark_models/face_landmarker.task.part landmark_models/face_landmarker.task
fi

"$ENV_BIN/python" -I -c "import seamless_interaction.fs, cv2, numpy; print('python ok')"
"$ENV_BIN/python" -I -c "import mediapipe as mp; mp.tasks.vision.FaceLandmarker.create_from_model_path('landmark_models/face_landmarker.task').close(); print('mediapipe + face_landmarker.task ok')"
"$ENV_BIN/python" -I -c "import face_alignment, torch; print('face_alignment ok, torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu only')"
"$ENV_BIN/eventcv" simulate --help > /dev/null && echo "eventcv ok"
"$ENV_BIN/ffmpeg" -hide_banner -encoders 2>/dev/null > /tmp/face_events_encoders.txt
grep -q libx264 /tmp/face_events_encoders.txt && echo "ffmpeg+libx264 ok" || { echo "ffmpeg lacks libx264" >&2; exit 1; }
rm -f /tmp/face_events_encoders.txt
echo "Setup done. Run: conda activate face_events"
