"""Generate dense face landmarks for every face crop already in the output dataset.

Walks a folder (default: OUT_DIR, as in pipeline.sh) for <id>_face<size>.mp4 crops and
writes only <id>_face<size>_landmarks.npz next to each (see generate_landmarks.py); no
videos are rendered. Clips that already have landmarks from the same method are skipped,
so it is resumable: stop it any time and rerun.

Usage:
  python scripts/landmarks_all.py [ROOT] [--method mediapipe|fa] [--workers N] [--overwrite]
  python scripts/landmarks_all.py ~/datasets/seamless_interaction_faces/naturalistic/dev --done-only

Env: OUT_DIR (default root), LANDMARK_MODEL, LANDMARKS_METHOD, LANDMARKS_WORKERS,
LANDMARKS_DEVICE=cpu|gpu (see generate_landmarks.py: CPU workers are faster overall).
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_landmarks import DEFAULT_MODEL, METHODS, generate, landmarks_path  # noqa: E402

CROP = re.compile(r"_face\d+$")  # the crop itself, not _kpts/_overlay/_render previews


def default_workers(method: str) -> int:
    """mediapipe runs on the CPU: one clip per worker. fa holds a model per worker on the GPU."""
    if os.environ.get("LANDMARKS_WORKERS"):
        return max(1, int(os.environ["LANDMARKS_WORKERS"]))
    return 1 if method == "fa" else max(1, min(8, (os.cpu_count() or 4) // 2))


def has_landmarks(video: Path, method: str) -> bool:
    out = landmarks_path(video)
    if not out.exists():
        return False
    try:
        return str(np.load(out)["method"]) == method
    except Exception:  # unreadable (e.g. truncated): redo it
        return False


def find_crops(root: Path, done_only: bool = False) -> list[Path]:
    crops = sorted(v for v in root.rglob("*_face*.mp4") if CROP.search(v.stem))
    if done_only:  # skip archives the pipeline may still be writing
        crops = [v for v in crops if (v.parent / "_DONE").exists()]
    return crops


def _one(video: Path, method: str, model: Path) -> None:
    generate(video, method, model, device=os.environ.get("LANDMARKS_DEVICE"))


def landmarks_for_all(videos: list[Path], method: str = "mediapipe", model: Path = DEFAULT_MODEL,
                      workers: int | None = None, log=print) -> list[str]:
    """Landmarks for several clips at once; returns the stems that failed."""
    workers = min(workers or default_workers(method), len(videos)) if videos else 0
    failed = []
    if workers <= 1:
        for video in videos:
            try:
                _one(video, method, model)
            except Exception as e:
                log(f"FAILED landmarks {video.stem}: {e}")
                failed.append(video.stem)
        return failed
    # spawn, not fork: the parent may hold threads (or a GPU context) that a fork would copy
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_one, v, method, model): v for v in videos}
        for future in as_completed(futures):
            video = futures[future]
            try:
                future.result()
            except Exception as e:
                log(f"FAILED landmarks {video.stem}: {e}")
                failed.append(video.stem)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = os.environ.get("OUT_DIR", "~/datasets/seamless_interaction_faces")
    ap.add_argument("root", type=Path, nargs="?", default=Path(default_root),
                    help=f"dataset output folder, searched recursively (default: {default_root})")
    ap.add_argument("--method", choices=METHODS, default=os.environ.get("LANDMARKS_METHOD", "mediapipe"))
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="mediapipe .task model (env LANDMARK_MODEL)")
    ap.add_argument("--workers", type=int, help="clips in parallel (env LANDMARKS_WORKERS)")
    ap.add_argument("--overwrite", action="store_true", help="redo clips that already have landmarks")
    ap.add_argument("--done-only", action="store_true",
                    help="only finished archives (_DONE): safe while pipeline.sh is running")
    ap.add_argument("--limit", type=int, help="at most N clips (testing)")
    args = ap.parse_args()
    root = args.root.expanduser()
    if not root.is_dir():
        ap.error(f"not a folder: {root}")
    if args.method == "mediapipe" and not args.model.exists():
        ap.error(f"{args.model} not found (run setup.sh, or pass --model)")

    crops = find_crops(root, args.done_only)
    todo = crops if args.overwrite else [v for v in crops if not has_landmarks(v, args.method)]
    todo = todo[: args.limit or None]
    workers = args.workers or default_workers(args.method)
    print(f"{len(crops)} crops under {root}, {len(todo)} to do ({args.method}, {workers} workers)",
          flush=True)
    t0 = time.time()
    failed = landmarks_for_all(todo, args.method, args.model, workers)
    print(f"done: {len(todo) - len(failed)} clips in {time.time() - t0:.0f}s, {len(failed)} failed",
          flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
