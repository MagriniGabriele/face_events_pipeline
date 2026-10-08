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

## Quick test (one archive, short events)

```bash
RAW_DIR=~/si_test_raw OUT_DIR=~/si_test_out EVENTS=1 EVENTS_MAX_FRAMES=150 \
  scripts/pipeline.sh --label naturalistic --split dev --batches 0 --limit 1
tail ~/si_test_out/pipeline.log
```

`EVENTS_MAX_FRAMES=150` limits events to the first 5 s of each clip (testing only).

## Full run (detached, survives closing the terminal)

```bash
conda activate face_events
EVENTS=1 setsid nohup scripts/pipeline.sh > /dev/null 2>&1 &
tail -f ~/datasets/seamless_interaction_faces/pipeline.log
```

Select what to process like the dataset is organised (as the repo's own batch downloads do):

```bash
scripts/pipeline.sh --label naturalistic --split train --batches 0-99   # batches 0..99
scripts/pipeline.sh --label improvised --split dev --batches 0,1          # list / ranges: "3", "0,5,10-12"
```

A *batch* is a dataset folder of ~40–50 archives (~50–100 GB); the pipeline still downloads
and processes **one archive (~1 GB) at a time** inside it, so disk use stays small. Without
filters it processes everything. `--limit N` additionally caps the number of archives per run
(handy for tests). The output keeps the dataset layout: `$OUT_DIR/{label}/{split}/{batch}/{archive}/`.
Stop with `pkill -f pipeline.sh`; rerun to resume. Failed archives are listed in
`$OUT_DIR/failed.txt` (their raw files are kept for inspection).

**Budget before a full run:** about 21k archives / 129k clips (~20 TB download). Per archive:
~5–7 min download + crop, and ~15 min / ~6 GB for full-length 640×640 events. Events for the
whole dataset at these settings are on the order of 100+ TB, so use `--label/--split/--limit`
or `EVENTS_SCALE` to fit your disk. The run stops when free space drops below `MIN_FREE_GB`.

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OUT_DIR` | `~/datasets/seamless_interaction_faces` | kept outputs |
| `RAW_DIR` | `~/datasets/seamless_interaction_raw` | temporary downloads (deleted per archive) |
| `CROP_SIZE` | `640` | crop side in original pixels (no resizing) |
| `EVENTS` | unset | `1` = simulate events |
| `EVENTS_SCALE` | unset | `"W H"`: events at this resolution; keypoints are remapped |
| `EVENTS_MAX_FRAMES` | unset | testing only: cap frames per clip |
| `EVENTCV` | `eventcv` on PATH | simulator binary |
| `MIN_FREE_GB` | `20` | abort when free disk space is below this |
| `SI_REPO` | `seamless_interaction` | path to the Meta repo clone |

## Output

`$OUT_DIR/{label}/{split}/{batch}/{archive}/`, per clip `<id>`:

| File | Content |
|---|---|
| `<id>_face640.mp4` | static 640×640 face crop (H.264, no audio) |
| `<id>_face640.npz` | keypoints/boxes shifted into crop coordinates (same keys as the original) |
| `<id>_face640.json` | crop position `crop_xywh` in the original frame, face extent, event stats |
| `<id>_face640_events.h5` | events: `events/{t (µs), x, y, p}`, attrs `width`, `height` |
| `<id>_face640_events_kpts.npz` | only with `EVENTS_SCALE`: keypoints at the event resolution |
| `<id>.npz`, `<id>.json` | original annotations (movement, SMPL-H, keypoints; transcript, VAD) |
| `_DONE` | archive finished (lists skipped clips, e.g. no detected face) |

Original `.mp4` and `.wav` are deleted. Crop pixel (x, y) = original pixel
(x + left, y + top), with `left, top` from `crop_xywh`. Without `EVENTS_SCALE`, events share
the crop's coordinates, so `<id>_face640.npz` applies to them directly.

Keypoints are COCO-WholeBody (133 × (x, y, score)): 0–16 body, 17–22 feet, **23–90 face
(68)**, 91–111 left hand, 112–132 right hand. They are snapped to a coarse ~15 px grid
(whole-body heatmap), so face landmarks are approximate.

## Inspecting results

```bash
# keypoints on a crop (window, q to quit)
python scripts/visualize_keypoints.py OUT/.../<id>_face640.mp4 --show
# events -> preview video
eventcv render OUT/.../<id>_face640_events.h5 preview.mp4 --dt-ms 33.333 --fps 30
```

## Files

- `setup.sh`, `environment.yml`: environment
- `scripts/pipeline.sh`: loop over all archives (resume, logging, disk check)
- `scripts/process_archive.py`: one archive: download → crop → events → cleanup
- `scripts/crop_faces.py`: face crop for a clip or a folder (also usable on its own)
- `scripts/visualize_keypoints.py`: draw keypoints on a video
- `video_to_events/run.sh`: EventCV settings (noise off, 8× upsampling); extra flags pass through
