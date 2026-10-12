"""Process one dataset archive: download -> face crop -> (events) -> delete raw.

Output goes to {out}/{label}/{split}/{batch:04d}/{archive:04d}/ with, per clip, the
face crop (.mp4/.npz/.json from crop_faces.py) plus the ORIGINAL .npz/.json
annotations. Original .mp4/.wav are not kept. A `_DONE` marker is written only after
every clip succeeded and the raw folder was deleted.

Usage:
  python scripts/process_archive.py improvised dev 0 0 \\
      --raw ~/datasets/seamless_interaction_raw --out ~/datasets/seamless_interaction_faces
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from crop_faces import crop_clip  # noqa: E402

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS  # noqa: E402


REPO = Path(__file__).resolve().parent.parent


def generate_events(crop_mp4: Path, out_dir: Path) -> Path:
    """Simulate events for one face crop -> <crop stem>_events.h5 (events/{t,x,y,p}).

    Same settings as video_to_events/run.sh (noise off, 8x upsample, eventcv defaults
    otherwise), via the Python API so it can run on the GPU. Env: EVENTS_DEVICE=gpu|cpu
    (default: gpu if available), EVENTS_SCALE="W H" (resample to the sensor resolution),
    EVENTS_MAX_FRAMES=N (testing only). Without EVENTS_SCALE the events keep the crop
    resolution, so crop keypoints are already in event coords.
    """
    import eventcv

    out = out_dir / f"{crop_mp4.stem}_events.h5"
    kw = {}
    if os.environ.get("EVENTS_SCALE"):
        kw["scale"] = tuple(int(v) for v in os.environ["EVENTS_SCALE"].split())
    if os.environ.get("EVENTS_MAX_FRAMES"):
        kw["max_frames"] = int(os.environ["EVENTS_MAX_FRAMES"])
    device = os.environ.get("EVENTS_DEVICE") or ("gpu" if eventcv.gpu_available() else "cpu")
    tmp = out.with_suffix(".tmp.h5")
    eventcv.simulate(str(crop_mp4), out=str(tmp), device=device, shot_noise_rate_hz=0,
                     leak_rate_hz=0, upsample="8", compression=1, **kw)
    tmp.rename(out)  # atomic: a partial file is never mistaken for a finished one
    print(f"  events on {device}")
    return out


def remap_keypoints(crop_npz: Path, events_resolution: tuple[int, int] | None = None) -> Path | None:
    """Scale crop-coordinate keypoints/box to the events resolution -> *_events_kpts.npz.

    Identity (returns None) when events_resolution is None or equals the crop size.
    """
    if events_resolution is None:
        return None
    data = np.load(crop_npz)
    sx = events_resolution[0] / args_size(crop_npz)
    sy = events_resolution[1] / args_size(crop_npz)
    out = crop_npz.with_name(f"{crop_npz.stem}_events_kpts.npz")
    kpts = data["boxes_and_keypoints:keypoints"].copy()
    kpts[..., 0] *= sx
    kpts[..., 1] *= sy
    np.savez_compressed(out, **{
        "boxes_and_keypoints:keypoints": kpts,
        "boxes_and_keypoints:box": data["boxes_and_keypoints:box"] * np.array([sx, sy, sx, sy]),
        "boxes_and_keypoints:is_valid_box": data["boxes_and_keypoints:is_valid_box"],
    })
    return out


def args_size(crop_npz: Path) -> int:
    return json.loads(crop_npz.with_suffix(".json").read_text())["crop_xywh"][2]


def events_step(crop: Path, out_dir: Path) -> None:
    t0 = time.time()
    ev = generate_events(crop, out_dir)
    meta_path = crop.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    scale = os.environ.get("EVENTS_SCALE")
    res = tuple(int(v) for v in scale.split()) if scale else None
    remapped = remap_keypoints(crop.with_suffix(".npz"), res)
    meta["events"] = {"file": ev.name, "resolution": list(res) if res else meta["crop_xywh"][2:],
                      "kpts_remapped": "identity" if remapped is None else remapped.name,
                      "max_frames": os.environ.get("EVENTS_MAX_FRAMES"),
                      "seconds": round(time.time() - t0, 1), "mb": round(ev.stat().st_size / 1e6, 1)}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  events {ev.name}: {meta['events']['mb']} MB in {meta['events']['seconds']}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("split")
    p.add_argument("batch", type=int)
    p.add_argument("archive", type=int)
    p.add_argument("--raw", type=Path, default=Path("~/datasets/seamless_interaction_raw"))
    p.add_argument("--out", type=Path, default=Path("~/datasets/seamless_interaction_faces"))
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--events", action="store_true", default=bool(os.environ.get("EVENTS")),
                   help="also simulate events per crop (env EVENTS=1)")
    p.add_argument("--events-only", action="store_true",
                   help="only add missing events to an already-cropped output folder")
    args = p.parse_args()
    raw, out_root = args.raw.expanduser(), args.out.expanduser()

    rel = Path(args.label, args.split, f"{args.batch:04d}")
    raw_dir = raw / rel / f"{args.archive:04d}"
    out_dir = out_root / rel / f"{args.archive:04d}"
    if args.events_only:
        for crop in sorted(out_dir.glob(f"*_face{args.size}.mp4")):
            if not crop.with_name(f"{crop.stem}_events.h5").exists():
                events_step(crop, out_dir)
        return
    if (out_dir / "_DONE").exists():
        print(f"already done: {out_dir}")
        return

    # a leftover tar means a previous extraction may have been interrupted
    if (raw / rel / f"{args.archive:04d}.tar").exists():
        shutil.rmtree(raw_dir, ignore_errors=True)
    fs = SeamlessInteractionFS(config=DatasetConfig(
        label=args.label, split=args.split, local_dir=str(raw)))
    ok, path = fs.download_archive_from_hf(
        args.archive, label=args.label, split=args.split, batch=args.batch, extract=True)
    if not ok:
        sys.exit(f"download/extract failed: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(raw_dir.rglob("*.mp4"))
    if not videos:
        sys.exit(f"no clips found in {raw_dir}")
    failed, skipped = [], []
    for video in videos:
        try:
            crop = crop_clip(video, out_dir, args.size)
            if crop is None:  # no annotations / no valid face: not retryable, just skip
                skipped.append(video.stem)
                continue
            for ext in (".npz", ".json"):  # originals may be read-only: replace, don't keep mode
                dst = out_dir / f"{video.stem}{ext}"
                dst.unlink(missing_ok=True)
                shutil.copyfile(video.with_suffix(ext), dst)
            if args.events:
                events_step(crop, out_dir)
        except Exception as e:
            print(f"FAILED {video.stem}: {e}", file=sys.stderr)
            failed.append(video.stem)
    if failed:
        sys.exit(f"{len(failed)}/{len(videos)} clips failed; raw kept in {raw_dir}")

    shutil.rmtree(raw_dir)
    (out_dir / "_DONE").write_text(
        f"{len(videos) - len(skipped)} clips\n" + "".join(f"skipped {s}\n" for s in skipped))
    print(f"done: {len(videos) - len(skipped)} clips ({len(skipped)} skipped) -> {out_dir}")


if __name__ == "__main__":
    main()
