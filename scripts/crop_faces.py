"""Crop a fixed (static) --size x --size face window per clip, for event simulation.

The window is computed once per clip from the COCO-WholeBody face keypoints
(indices 23-90) in the matching .npz: per-frame face boxes are extended upward to
include forehead/hair, a robust union is taken over the clip (percentiles, to
ignore outliers), padded, and a window of exactly --size pixels is centered on it
and clamped to the frame. No resizing: crop pixels are original pixels. A static
window avoids spurious events from a moving crop.

Alongside each crop, a .npz holds the keypoints/boxes shifted into crop coordinates.

Usage:
  python scripts/crop_faces.py ~/datasets/seamless_interaction/naturalistic -o face_crops --size 640
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

FACE = slice(23, 91)


def face_crop_box(data, frame_w: int, frame_h: int, size: int, margin: float, min_score: float):
    kpts = data["boxes_and_keypoints:keypoints"][:, FACE]
    valid = data["boxes_and_keypoints:is_valid_box"] & (kpts[:, :, 2].mean(1) >= min_score)
    if valid.sum() == 0:
        return None
    pts = kpts[valid, :, :2]
    lo, hi = pts.min(1), pts.max(1)
    # 68 landmarks span brows to chin: extend upward by half the face height
    lo[:, 1] -= 0.5 * (hi[:, 1] - lo[:, 1])
    x0, y0 = np.percentile(lo, 2, axis=0)
    x1, y1 = np.percentile(hi, 98, axis=0)

    needed = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - size / 2, 0, frame_w - size))
    top = int(np.clip(cy - size / 2, 0, frame_h - size))
    return left, top, int(needed), float(valid.mean())


def probe(video: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        text=True,
    )
    w, h = out.split()[0].split(",")[:2]
    return int(w), int(h)


def crop_clip(video: Path, out_dir: Path, size: int, margin: float = 0.1,
              min_score: float = 0.5, crf: int = 18) -> Path | None:
    """Crop one clip; returns the output mp4 path, or None if skipped (no npz / no face)."""
    npz = video.with_suffix(".npz")
    if not npz.exists():
        print(f"skip {video.name}: no .npz", flush=True)
        return None
    w, h = probe(video)
    data = np.load(npz)
    box = face_crop_box(data, w, h, size, margin, min_score)
    if box is None:
        print(f"skip {video.name}: no valid face keypoints", flush=True)
        return None
    left, top, needed, valid_frac = box
    out = out_dir / f"{video.stem}_face{size}.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"crop={size}:{size}:{left}:{top}",
         "-an", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    offset = np.array([left, top], dtype=np.float32)
    kpts = data["boxes_and_keypoints:keypoints"].copy()
    kpts[..., :2] -= offset
    np.savez_compressed(
        out.with_suffix(".npz"),
        **{
            "boxes_and_keypoints:keypoints": kpts,
            "boxes_and_keypoints:box": data["boxes_and_keypoints:box"] - np.tile(offset, 2),
            "boxes_and_keypoints:is_valid_box": data["boxes_and_keypoints:is_valid_box"],
        },
    )
    fits = needed <= size
    meta = {"source": str(video), "crop_xywh": [left, top, size, size],
            "face_extent": needed, "fits": fits, "valid_frac": round(valid_frac, 3)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    warn = "" if fits else "  WARNING: face extent exceeds crop"
    print(f"  face extent {needed}px, crop {size}x{size} at ({left},{top}){warn}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="an .mp4 or a directory searched recursively")
    p.add_argument("-o", "--output", type=Path, default=Path("face_crops"))
    p.add_argument("--size", type=int, default=640, help="crop side in original pixels")
    p.add_argument("--margin", type=float, default=0.1, help="padding on each side, fraction of box")
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--crf", type=int, default=18)
    args = p.parse_args()

    videos = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.mp4"))
    args.output.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.stem}", flush=True)
        crop_clip(video, args.output, args.size, args.margin, args.min_score, args.crf)


if __name__ == "__main__":
    main()
