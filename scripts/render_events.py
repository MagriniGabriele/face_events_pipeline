"""Render an events .h5 to .mp4 (ON red, OFF blue) with the face keypoints drawn on top.

One output frame per video frame (30 fps by default): frame i shows the events with
t in [i/fps, (i+1)/fps) and keypoint row i. Events are streamed in chunks, so full
clips (hundreds of millions of events) do not need to fit in memory.

Keypoints default to <id>_face640_events_kpts.npz (when events were rescaled) or
<id>_face640.npz (same coordinates as the events).

Usage:
  python scripts/render_events.py OUT/.../<id>_face640_events.h5 [-o out.mp4] [--all] [--show]
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np

FACE = range(23, 91)
GROUPS = {
    "body": (range(0, 23), (0, 255, 0)),
    "face": (FACE, (255, 255, 255)),
    "lhand": (range(91, 112), (0, 200, 255)),
    "rhand": (range(112, 133), (255, 0, 255)),
}
CHUNK = 5_000_000


def default_npz(h5: Path) -> Path | None:
    base = h5.name.removesuffix("_events.h5")
    for name in (f"{base}_events_kpts.npz", f"{base}.npz"):
        if (h5.parent / name).exists():
            return h5.parent / name
    return None


def frames(ev, w: int, h: int, dt_us: float, max_frames: int | None):
    """Yield (index, BGR image) per time window, streaming the event arrays in chunks."""
    n = len(ev["t"])
    img = np.zeros((h, w, 3), np.uint8)
    cur = 0
    for s in range(0, n, CHUNK):
        t = ev["t"][s:s + CHUNK]
        x, y, p = ev["x"][s:s + CHUNK], ev["y"][s:s + CHUNK], ev["p"][s:s + CHUNK].astype(bool)
        idx = (t // dt_us).astype(np.int64)  # sorted, since t is
        uniq, starts = np.unique(idx, return_index=True)
        ends = np.append(starts[1:], len(idx))
        for f, a, b in zip(uniq, starts, ends):
            while cur < f:  # emit finished (or empty) frames
                yield cur, img
                img = np.zeros((h, w, 3), np.uint8)
                cur += 1
                if max_frames and cur >= max_frames:
                    return
            xs, ys, ps = x[a:b], y[a:b], p[a:b]
            img[ys[ps], xs[ps]] = (0, 0, 255)    # ON: red
            img[ys[~ps], xs[~ps]] = (255, 0, 0)  # OFF: blue
    yield cur, img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("events", type=Path,
                    help="<id>_face640_events.h5 (or the clip's _face640.mp4/.npz: the .h5 is found next to it)")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--npz", type=Path, help="keypoints (default: next to the .h5)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--all", action="store_true", help="draw body and hands too")
    ap.add_argument("--no-kpts", action="store_true", help="events only")
    ap.add_argument("--show", action="store_true", help="show live window (q to quit)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--crf", type=int, default=32,
                    help="H.264 quality: lower = better/bigger (event frames compress poorly; 32 looks fine)")
    args = ap.parse_args()
    if args.events.suffix != ".h5":  # given the crop .mp4/.npz: use its events file
        args.events = args.events.with_name(args.events.stem.removesuffix("_events") + "_events.h5")
    if not args.events.exists():
        ap.error(f"events file not found: {args.events}")

    f = h5py.File(args.events, "r")
    ev, w, h = f["events"], int(f.attrs["width"]), int(f.attrs["height"])
    dt_us = 1e6 / args.fps / float(f.attrs.get("timestamp_scale_ms", 0.001) * 1000)

    kpts = None
    npz = args.npz or default_npz(args.events)
    if npz and not args.no_kpts:
        data = np.load(npz)
        kpts, valid = data["boxes_and_keypoints:keypoints"], data["boxes_and_keypoints:is_valid_box"]
        print(f"keypoints: {npz.name}")
    groups = GROUPS if args.all else {"face": GROUPS["face"]}

    out = args.output or args.events.with_name(args.events.stem + "_render.mp4")
    writer = None
    if args.output or not args.show:
        writer = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
             str(out)],
            stdin=subprocess.PIPE,
        )

    n = 0
    for i, img in frames(ev, w, h, dt_us, args.max_frames):
        if kpts is not None and i < len(kpts) and valid[i]:
            for idx, color in groups.values():
                for x, y, s in kpts[i, list(idx)]:
                    if s >= args.min_score:
                        cv2.circle(img, (int(x), int(y)), 2, color, -1)
        cv2.putText(img, f"frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if writer:
            writer.stdin.write(img.tobytes())
        if args.show:
            cv2.imshow("events", img)
            if cv2.waitKey(int(1000 / args.fps)) & 0xFF == ord("q"):
                break
        n += 1

    if writer:
        writer.stdin.close()
        writer.wait()
        print(f"wrote {out} ({n} frames)")
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
