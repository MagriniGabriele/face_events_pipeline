"""Process one dataset archive: download -> face crop -> (events) -> (landmarks) -> delete raw.

Output goes to {out}/{label}/{split}/{batch:04d}/{archive:04d}/ with, per clip, the
face crop (.mp4/.npz/.json from crop_faces.py) plus the ORIGINAL .npz/.json
annotations. Original .mp4/.wav are not kept. A `_DONE` marker is written only after
every clip succeeded and the raw folder was deleted.

Usage:
  python scripts/process_archive.py improvised dev 0 0 \\
      --raw ~/datasets/seamless_interaction_raw --out ~/datasets/seamless_interaction_faces

`--landmarks` (env LANDMARKS=1) adds <id>_face<size>_landmarks.npz per crop as the last
step (scripts/generate_landmarks.py; method from LANDMARKS_METHOD, default mediapipe).

`--max-clips N` stops after N clips (smoke tests): the archive is then *not* marked
`_DONE` and the raw folder is kept, so a later full run finishes it.
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from crop_faces import crop_clip  # noqa: E402
from landmarks_all import landmarks_for_all  # noqa: E402

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS  # noqa: E402


REPO = Path(__file__).resolve().parent.parent


def ts() -> str:
    """Timestamp for progress lines, so a stalled step is visible in the log."""
    return time.strftime("%F %T")


def say(line: str, err: bool = False) -> None:
    """Print one line in a single write.

    The worker pool has several processes writing to the same log, and print() emits the
    text and its newline as two writes, which lets two workers land on one line.
    """
    stream = sys.stderr if err else sys.stdout
    stream.write(line + "\n")
    stream.flush()


PROGRESS = re.compile(r"(\d+)/(\d+) frames, ([\d.]+[KMG]?) events")


def bar(done: int, total: int, width: int = 14) -> str:
    filled = round(width * done / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + f"] {100 * done // max(total, 1):3d}%"


@contextlib.contextmanager
def clip_progress(label: str, every: float | None = None):
    """Report one progress line per clip every `every` seconds (EVENTS_PROGRESS_SECS).

    eventcv prints `simulating: 166/180 frames, 13.6M events` to stderr, overwriting
    itself with \\r: unreadable in a log file and hopeless with several workers writing
    at once. So stderr is redirected to a scratch file for the duration of the call and
    sampled on a timer, which turns it into one timestamped line per clip per interval.
    """
    if every is None:
        every = float(os.environ.get("EVENTS_PROGRESS_SECS", 30))
    scratch = Path(tempfile.mkstemp(prefix="evprog_", suffix=".txt")[1])
    saved = os.dup(2)  # fd level: the progress comes from native code, not sys.stderr
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(every):
            text = scratch.read_text(errors="replace")
            found = PROGRESS.findall(text)
            if not found:
                continue
            done, total, events = found[-1]
            say(f"{ts()}   {label} {bar(int(done), int(total))} "
                f"{done}/{total} frames, {events} events")

    with open(scratch, "wb") as sink:
        os.dup2(sink.fileno(), 2)
    watcher = threading.Thread(target=report, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join(timeout=1)
        os.dup2(saved, 2)
        os.close(saved)
        # anything that was not a progress update (warnings, a traceback) must not vanish
        rest = PROGRESS.sub("", scratch.read_text(errors="replace").replace("\r", "\n"))
        rest = "\n".join(line for line in rest.splitlines() if line.strip(" :simulating,"))
        if rest.strip():
            say(rest, err=True)
        scratch.unlink(missing_ok=True)


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
    # ~3 min per full-length clip, so announce before starting and report a bar while it runs
    say(f"{ts()}   events on {device}: {out.name} ...")
    # compression: left at eventcv's .h5 default, gzip level 1. Measured on one 4090:
    # it costs ~12% throughput with a full worker pool (the gzip runs on idle cores)
    # and saves 4.2x the disk — 1.9 GB instead of 8 GB per full-length clip.
    tmp.unlink(missing_ok=True)  # a leftover from a killed run would just waste disk
    try:
        with clip_progress(crop_mp4.stem):
            eventcv.simulate(str(crop_mp4), out=str(tmp), device=device, shot_noise_rate_hz=0,
                             leak_rate_hz=0, upsample="8", progress=True, **kw)
    except BaseException:
        tmp.unlink(missing_ok=True)  # else a failed clip leaves GBs of partial .tmp.h5
        raise
    tmp.rename(out)  # atomic: a partial file is never mistaken for a finished one
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
    say(f"{ts()}   events {ev.name}: {meta['events']['mb']} MB in "
        f"{meta['events']['seconds']}s")


def events_pool_size() -> int:
    """Concurrent clips for the events step (env EVENTS_WORKERS, default 6).

    A single clip leaves the GPU at ~46% busy, so several at once nearly triple the
    throughput (measured on one RTX 4090 at 640x640: 31 fps for one worker, 87 for six,
    91 for eight); each worker's h5 gzip also gets its own core. Past ~6 the GPU is the
    wall. 1 = sequential, the old behaviour.
    """
    return max(1, int(os.environ.get("EVENTS_WORKERS", 6)))


def events_for_all(crops: list[Path], out_dir: Path) -> list[str]:
    """Run the events step for several crops at once; returns the stems that failed."""
    workers = min(events_pool_size(), len(crops))
    if workers <= 1:
        failed = []
        for crop in crops:
            try:
                events_step(crop, out_dir)
            except Exception as e:
                say(f"FAILED events {crop.stem}: {e}", err=True)
                failed.append(crop.stem)
        return failed

    say(f"{ts()} events: {len(crops)} clips, {workers} workers")
    failed = []
    # fork is fine here: eventcv (and the GPU context) is imported inside the children
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(events_step, crop, out_dir): crop for crop in crops}
        for future in as_completed(futures):
            crop = futures[future]
            try:
                future.result()
            except Exception as e:
                say(f"FAILED events {crop.stem}: {e}", err=True)
                failed.append(crop.stem)
    return failed


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
    p.add_argument("--landmarks", action="store_true", default=bool(os.environ.get("LANDMARKS")),
                   help="also generate dense face landmarks per crop, as the last step (env LANDMARKS=1)")
    p.add_argument("--events-only", action="store_true",
                   help="only add missing events to an already-cropped output folder")
    p.add_argument("--max-clips", type=int, default=0, metavar="N",
                   help="stop after N clips (smoke test): no _DONE, raw folder kept")
    args = p.parse_args()
    raw, out_root = args.raw.expanduser(), args.out.expanduser()

    rel = Path(args.label, args.split, f"{args.batch:04d}")
    raw_dir = raw / rel / f"{args.archive:04d}"
    out_dir = out_root / rel / f"{args.archive:04d}"
    if args.events_only:
        crops = sorted(out_dir.glob(f"*_face{args.size}.mp4"))[: args.max_clips or None]
        todo = [c for c in crops if not c.with_name(f"{c.stem}_events.h5").exists()]
        if events_for_all(todo, out_dir):
            sys.exit("some clips failed")
        return
    if (out_dir / "_DONE").exists():
        say(f"already done: {out_dir}")
        return

    # a leftover tar means a previous extraction may have been interrupted
    tar = raw / rel / f"{args.archive:04d}.tar"
    if tar.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)
    # No tar but clips on disk = a finished extraction (the tar is removed only after
    # extractall returns), so resume from it. download_archive_from_hf would otherwise
    # re-download the whole ~1 GB tar before noticing the extract dir, and keep it.
    if any(raw_dir.rglob("*.mp4")):
        say(f"{ts()} resuming from extracted {raw_dir}")
    else:
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
    total = len(videos)
    truncated = bool(args.max_clips) and total > args.max_clips
    if truncated:
        videos = videos[: args.max_clips]
    failed, skipped, crops = [], [], []
    for i, video in enumerate(videos, 1):
        say(f"{ts()} [{i}/{len(videos)}] {video.stem}")
        try:
            crop = crop_clip(video, out_dir, args.size)
            if crop is None:  # no annotations / no valid face: not retryable, just skip
                skipped.append(video.stem)
                continue
            for ext in (".npz", ".json"):  # originals may be read-only: replace, don't keep mode
                dst = out_dir / f"{video.stem}{ext}"
                dst.unlink(missing_ok=True)
                shutil.copyfile(video.with_suffix(ext), dst)
            crops.append(crop)
        except Exception as e:
            say(f"FAILED {video.stem}: {e}", err=True)
            failed.append(video.stem)
    # events after every crop of this archive exists, so the whole pool stays busy
    if args.events and crops:
        failed += events_for_all(crops, out_dir)
    if args.landmarks and crops:
        method = os.environ.get("LANDMARKS_METHOD", "mediapipe")
        say(f"{ts()} landmarks: {len(crops)} clips ({method})")
        failed += landmarks_for_all(crops, method, log=lambda m: say(m, err=True))
    if failed:
        sys.exit(f"{len(failed)}/{len(videos)} clips failed; raw kept in {raw_dir}")

    kept = len(videos) - len(skipped)
    if truncated:  # an incomplete archive must not look finished to the next run
        say(f"partial: {kept} of {total} clips (--max-clips {args.max_clips}), "
            f"no _DONE written, raw kept in {raw_dir}")
        return
    shutil.rmtree(raw_dir)
    tar.unlink(missing_ok=True)  # only present if a resume re-downloaded it unused
    (out_dir / "_DONE").write_text(
        f"{kept} clips\n" + "".join(f"skipped {s}\n" for s in skipped))
    say(f"done: {kept} clips ({len(skipped)} skipped) -> {out_dir}")


if __name__ == "__main__":
    main()
