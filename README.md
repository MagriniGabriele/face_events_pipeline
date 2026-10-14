# Face events pipeline (Seamless Interaction)

Downloads the [Seamless Interaction](https://github.com/facebookresearch/seamless_interaction)
dataset one archive (~1 GB, ~12 clips) at a time, crops a static 640×640 face window per
clip, simulates events with EventCV, keeps the annotations, and deletes the original
videos before moving to the next archive. Resumable: rerun the same command and finished
archives are skipped.

## Setup (once per machine)

Needs `git` and `conda` (Miniconda/Miniforge). Linux, ~5 GB for the env.

```bash
cd face_events_pipeline
bash setup.sh              # clones facebookresearch/seamless_interaction here + creates env "face_events"
conda activate face_events
```

The env (`environment.yml`) has Python 3.12, ffmpeg (libx264), `eventcv==1.0.7`, h5py, and
the Meta repo installed in editable mode (its `assets/filelist.csv` lists every archive).

## Quick test (two clips, short events)

```bash
RAW_DIR=~/si_test_raw OUT_DIR=~/si_test_out EVENTS=1 EVENTS_MAX_FRAMES=150 \
  scripts/pipeline.sh --label naturalistic --split dev --batches 0 --limit 1 --max-clips 2
timeout 60 tail -f ~/si_test_out/pipeline.log      # progress appears per clip, live
```

`EVENTS_MAX_FRAMES=150` limits events to the first 5 s of each clip, `--max-clips 2` to
the first 2 clips of the archive (testing only). An archive processed with `--max-clips`
is deliberately left unfinished — no `_DONE`, raw files kept — so a later full run
completes it.

Everything the worker prints goes to `$OUT_DIR/pipeline.log`, not to the terminal: a clip
takes ~6 s to crop and ~3 min to simulate events, so watch the log with `tail -f`
rather than the terminal, which stays silent between archives. Each clip being simulated
reports its own completion bar every `EVENTS_PROGRESS_SECS` (default 30):

```
18:05:38   V00_S0644_..._face640 [##------------]  12% 113/900 frames, 8.6M events
18:05:38   V00_S0692_..._face640 [##------------]  12% 115/900 frames, 8.8M events
18:06:14   events V00_S0644_..._events.h5: 247.7 MB in 41.3s
```

## Full run

```bash
conda activate face_events
EVENTS=1 scripts/pipeline.sh --label naturalistic --split train --batches 0-99
```

Select what to process like the dataset is organised (as the repo's own batch downloads do):

```bash
scripts/pipeline.sh --label naturalistic --split train --batches 0-99   # batches 0..99
scripts/pipeline.sh --label improvised --split dev --batches 0,1          # list / ranges: "3", "0,5,10-12"
```

A *batch* is a dataset folder of ~40–50 archives (~50–100 GB); the pipeline still downloads
and processes **one archive (~1 GB) at a time** inside it, so disk use stays small. Without
filters it processes everything. `--limit N` caps the number of **archives** per run — not
clips: an archive holds 4–19 clips of ~3 min each, so `--limit 2` is ~20 clips. Use
`--max-clips N` to cap the clips *within* each archive (smoke tests). The output keeps the
dataset layout: `$OUT_DIR/{label}/{split}/{batch}/{archive}/`.
Stop with `pkill -f pipeline.sh`; rerun to resume. Failed archives are listed in
`$OUT_DIR/failed.txt` (their raw files are kept for inspection).

**Budget before a full run:** about 21k archives / 129k clips (~20 TB download). Measured on
one RTX 4090, per 3 min 1080×1920 clip: ~6 s to crop, and ~3 min / ~1.9 GB for full-length
640×640 events (~27 fps, ~600M events) on a single worker. Clips of an archive are simulated
`EVENTS_WORKERS` at a time, which nearly triples throughput (a single clip leaves the GPU
~46% idle): 31 fps for one worker, 87 for six, 91 for eight. Per archive that is ~2.5 min
without events and **~13–15 min / ~20–25 GB with them**. Events for the whole dataset at
these settings are on the order of 200 TB, so use `--label/--split/--limit/--max-clips` or
`EVENTS_SCALE` to fit your disk. The run stops when free space drops below `MIN_FREE_GB`
— checked *before* each archive, so keep it above the ~25 GB one archive can write.

The `.h5` keeps eventcv's default gzip-1: with a full worker pool it costs ~12% throughput
and saves 4.2× the disk (1.9 GB instead of 8 GB per clip).

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OUT_DIR` | `~/datasets/seamless_interaction_faces` | kept outputs |
| `RAW_DIR` | `~/datasets/seamless_interaction_raw` | temporary downloads (deleted per archive) |
| `CROP_SIZE` | `640` | crop side in original pixels (no resizing) |
| `EVENTS` | `1` | `1` = simulate events |
| `EVENTS_SCALE` | unset | `"W H"`: events at this resolution; keypoints are remapped |
| `EVENTS_MAX_FRAMES` | unset | testing only: cap frames per clip |
| `EVENTS_DEVICE` | `gpu` if available, else `cpu` | where eventcv simulates (`gpu` = compute shader via Vulkan; same events as CPU) |
| `EVENTS_WORKERS` | `6` | clips simulated in parallel per archive (`1` = sequential). eventcv exposes no GPU index, so all workers share GPU 0 even on a multi-GPU box |
| `EVENTS_PROGRESS_SECS` | `30` | seconds between per-clip progress bars in the log |
| `LANDMARKS` | `1` | `1` = dense face landmarks per crop, as the last step (see [Face landmarks](#face-landmarks)) |
| `LANDMARKS_METHOD` | `mediapipe` | `mediapipe` (478 pts) or `fa` (face_alignment, 68 pts) |
| `LANDMARKS_DEVICE` | mediapipe: `cpu`, fa: `gpu` if available | `cpu` or `gpu`; for mediapipe, CPU workers are faster over many clips and leave the GPU to eventcv |
| `LANDMARKS_WORKERS` | half the cores, max 8 (`fa`: 1) | clips processed in parallel for landmarks |
| `LANDMARK_MODEL` | `landmark_models/face_landmarker.task` | MediaPipe model (downloaded by `setup.sh`) |
| `MIN_FREE_GB` | `20` | abort when free disk space is below this |
| `SI_REPO` | `seamless_interaction` | path to the Meta repo clone |
| `FEP_PYTHON` | `python` on PATH | interpreter to use (the active env by default) |
| `HF_TOKEN` | unset | optional: a [HF access token](https://huggingface.co/settings/tokens) lifts the anonymous rate limit and speeds up downloads |

Downloads work without `HF_TOKEN` (the dataset is public) — you just get a warning per
archive and the anonymous rate limit. For a long run, `export HF_TOKEN=hf_...` first.

## Output

`$OUT_DIR/{label}/{split}/{batch}/{archive}/`, per clip `<id>`:

| File | Content |
|---|---|
| `<id>_face640.mp4` | static 640×640 face crop (H.264, no audio) |
| `<id>_face640.npz` | keypoints/boxes shifted into crop coordinates (same keys as the original) |
| `<id>_face640.json` | crop position `crop_xywh` in the original frame, face extent, event stats |
| `<id>_face640_events.h5` | events: `events/{t (µs), x, y, p}`, attrs `width`, `height` |
| `<id>_face640_events_kpts.npz` | only with `EVENTS_SCALE`: keypoints at the event resolution |
| `<id>_face640_landmarks.npz` | only with `LANDMARKS=1` or `landmarks_all.py`: dense face landmarks in crop pixels |
| `<id>.npz`, `<id>.json` | original annotations (movement, SMPL-H, keypoints; transcript, VAD) |
| `_DONE` | archive finished (lists skipped clips, e.g. no detected face) |

Original `.mp4` and `.wav` are deleted. Crop pixel (x, y) = original pixel
(x + left, y + top), with `left, top` from `crop_xywh`. Without `EVENTS_SCALE`, events share
the crop's coordinates, so `<id>_face640.npz` applies to them directly.

Keypoints are COCO-WholeBody (133 × (x, y, score)): 0–16 body, 17–22 feet, **23–90 face
(68)**, 91–111 left hand, 112–132 right hand. They are snapped to a coarse ~15 px grid
(whole-body heatmap), so face landmarks are approximate.

## Face landmarks

`scripts/generate_landmarks.py` writes `<id>_face640_landmarks.npz` (annotations only):
`landmarks` (T, N, 2) in crop pixels (NaN without a face), `valid` (T,), `width`, `height`,
`fps`, `method`; with MediaPipe also `z`, `blendshapes` (T, 52) + `blendshape_names`, and
`transform` (T, 4, 4, head pose). Two methods: `mediapipe` (default, 478 points tracked
across frames, ~130–150 fps per clip on CPU, ~215 with `--device gpu`) or `fa`
(face_alignment, 68 iBUG points, GPU).

```bash
LANDMARKS=1 EVENTS=1 scripts/pipeline.sh ...               # as the last step of every archive
python scripts/landmarks_all.py [OUT_DIR] [--done-only]    # for all crops already downloaded (resumable)
python scripts/generate_landmarks.py OUT/.../<id>_face640.mp4 [--method fa]   # one clip
```

`--done-only` skips archives still being written, so `landmarks_all.py` can run alongside
the pipeline. When the file exists, the scripts below draw these landmarks instead of the
coarse COCO face keypoints (rescaled to the event resolution with `EVENTS_SCALE`);
`--coco-face` brings the old ones back.

## Inspecting results

```bash
# keypoints on a crop (window, q to quit)
python scripts/visualize_keypoints.py OUT/.../<id>_face640.mp4 --show
# events + face keypoints -> <id>_face640_events_render.mp4 (add --show for a live window,
# --all for body/hands, --no-kpts for events only, --max-frames N for a quick look)
python scripts/render_events.py OUT/.../<id>_face640_events.h5
# RGB crop + events (rendered by eventcv) + keypoints -> <id>_face640_overlay.mp4
python scripts/overlay_events.py OUT/.../<id>_face640.mp4 [--mode side] [--max-frames 300]
```

## Files

- `setup.sh`, `environment.yml`: environment
- `scripts/pipeline.sh`: loop over all archives (resume, logging, disk check)
- `scripts/process_archive.py`: one archive: download → crop → events → cleanup
- `scripts/crop_faces.py`: face crop for a clip or a folder (also usable on its own)
- `scripts/visualize_keypoints.py`: draw keypoints on a video
- `scripts/overlay_events.py`: events (via `eventcv render`) over the RGB crop + face keypoints, or side by side (`--mode side`)
- `scripts/generate_landmarks.py`: dense face landmarks for a clip (MediaPipe or face_alignment)
- `scripts/landmarks_all.py`: landmarks for every crop in a dataset folder (parallel, resumable)
- `scripts/render_events.py`: render an events .h5 to .mp4 with keypoints overlaid (streams, works on full clips)
- `video_to_events/run.sh`: reference EventCV settings (noise off, 8× upsampling), CLI/CPU only; the pipeline uses the same settings through the Python API
