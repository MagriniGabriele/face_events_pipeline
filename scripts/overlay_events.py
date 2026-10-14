"""Overlay events (rendered by `eventcv render`) on the RGB face crop, plus face keypoints.

The face points come from <id>_face640_landmarks.npz (scripts/generate_landmarks.py)
when it exists, else from the COCO face keypoints in <id>_face640.npz (--coco-face
forces those).

Give any file of a clip (<id>_face640.mp4, .npz or _events.h5); the others are found
next to it. Writes <id>_face640_overlay.mp4.

  --mode overlay  events painted over the dimmed RGB (default)
  --mode side     RGB | events side by side

Usage:
  python scripts/overlay_events.py OUT/.../<id>_face640.mp4 [--mode side] [--no-kpts] [--max-frames N]
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from generate_landmarks import draw_landmarks, landmarks_path, load_landmarks

FACE = range(23, 91)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path, help="<id>_face640.mp4 / .npz / _events.h5")
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--mode", choices=["overlay", "side"], default="overlay")
    ap.add_argument("--dim", type=float, default=0.75, help="RGB brightness under the events (overlay)")
    ap.add_argument("--no-kpts", action="store_true")
    ap.add_argument("--landmarks", type=Path, help="dense face landmarks (default: next to the clip)")
    ap.add_argument("--coco-face", action="store_true", help="draw the COCO face keypoints, not the landmarks")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--crf", type=int, default=1)
    args = ap.parse_args()

    stem = args.clip.name.removesuffix("_events.h5").removesuffix(".mp4").removesuffix(".npz")
    rgb, h5, npz = (args.clip.with_name(stem + s) for s in (".mp4", "_events.h5", ".npz"))
    for f in (rgb, h5):
        if not f.exists():
            ap.error(f"not found: {f}")
    out = args.output or rgb.with_name(stem + "_overlay.mp4")

    cap = cv2.VideoCapture(str(rgb))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = Path(tempfile.mkdtemp())
    ev_mp4 = tmp / "events.mp4"
    cmd = [shutil.which("eventcv") or "eventcv", "render", str(h5), str(ev_mp4),
           "--dt-ms", f"{1000 / fps:.6f}", "--fps", str(fps), "-q"]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    print(f"[1/2] rendering events with eventcv ({total} frames, {total / fps / 60:.1f} min of video)...",
          flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"      done in {time.time() - t0:.0f}s", flush=True)
    evcap = cv2.VideoCapture(str(ev_mp4))

    kpts = lms = None
    lm_file = args.landmarks or landmarks_path(rgb)
    if not args.no_kpts and not args.coco_face and lm_file.exists():
        lms, lm_valid = load_landmarks(lm_file, w, h)
        print(f"face: {lm_file.name}")
    elif not args.no_kpts and npz.exists():
        data = np.load(npz)
        kpts, valid = data["boxes_and_keypoints:keypoints"], data["boxes_and_keypoints:is_valid_box"]

    ow = w * 2 if args.mode == "side" else w
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
         str(out)],
        stdin=subprocess.PIPE,
    )

    print("[2/2] combining with RGB + keypoints", flush=True)
    t0 = time.time()
    i = 0
    while args.max_frames is None or i < args.max_frames:
        ok, frame = cap.read()
        ok2, ev = evcap.read()
        if not ok:
            break
        if not ok2:  # events ended earlier (e.g. --max-frames at simulation): blank
            ev = np.zeros_like(frame)
        if ev.shape != frame.shape:
            ev = cv2.resize(ev, (w, h), interpolation=cv2.INTER_NEAREST)
        if args.mode == "overlay":
            mask = ev.max(axis=2) > 12  # black background, (dim) coloured events
            img = (frame * args.dim).astype(np.uint8)
            warm = ev[..., 2].astype(np.int16) >= ev[..., 0]  # eventcv: warm vs cool polarity colours
            img[mask & warm] = (0, 0, 255)     # red
            img[mask & ~warm] = (255, 255, 0)  # cyan
            panels = [img]
        else:
            panels = [frame, ev]
        if lms is not None and i < len(lms) and lm_valid[i]:
            for p in panels:
                draw_landmarks(p, lms[i], (0, 255, 0), 1)
        elif kpts is not None and i < len(kpts) and valid[i]:
            for p in panels:
                for x, y, s in kpts[i, list(FACE)]:
                    if s >= 0.3:
                        cv2.circle(p, (int(x), int(y)), 2, (0, 255, 0), -1)
        img = np.hstack(panels)
        cv2.putText(img, f"frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        writer.stdin.write(img.tobytes())
        i += 1
        if i % 30 == 0 or i == total:
            rate = i / (time.time() - t0)
            eta = (total - i) / rate if rate else 0
            print(f"\r      {i}/{total} frames ({100 * i / max(total, 1):.0f}%), "
                  f"{rate:.0f} fps, ~{eta / 60:.1f} min left   ", end="", flush=True)
    print()

    writer.stdin.close()
    writer.wait()
    shutil.rmtree(tmp)
    print(f"wrote {out} ({i} frames)")


if __name__ == "__main__":
    main()
