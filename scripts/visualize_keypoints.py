"""Overlay facial keypoints (and optionally body/hands) from a .npz onto its .mp4.

Keypoints follow the COCO-WholeBody 133-point layout, (x, y, score) in pixels:
  0-16 body, 17-22 feet, 23-90 face (68 pts), 91-111 left hand, 112-132 right hand.
The face is drawn from <stem>_landmarks.npz (scripts/generate_landmarks.py) when it
exists, else from the COCO face keypoints (--coco-face forces those).

Usage:
  python scripts/visualize_keypoints.py path/to/V00_..._P0069.mp4 [--npz ann.npz] [-o out.mp4] [--all] [--show]
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from generate_landmarks import draw_landmarks, landmarks_path, load_landmarks

FACE = range(23, 91)
GROUPS = {
    "body": (range(0, 23), (0, 255, 0)),
    "face": (FACE, (0, 0, 255)),
    "lhand": (range(91, 112), (255, 128, 0)),
    "rhand": (range(112, 133), (255, 0, 255)),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--npz", type=Path, help="annotation file (default: <video>.npz)")
    p.add_argument("--all", action="store_true", help="draw body and hands too")
    p.add_argument("--show", action="store_true", help="show live window (q to quit)")
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--landmarks", type=Path, help="dense face landmarks (default: <video>_landmarks.npz)")
    p.add_argument("--coco-face", action="store_true", help="draw the COCO face keypoints, not the landmarks")
    args = p.parse_args()

    data = np.load(args.npz or args.video.with_suffix(".npz"))
    kpts = data["boxes_and_keypoints:keypoints"]
    boxes = data["boxes_and_keypoints:box"]
    valid = data["boxes_and_keypoints:is_valid_box"]
    groups = GROUPS if args.all else {"face": GROUPS["face"]}

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lm_file = args.landmarks or landmarks_path(args.video)
    lms = None
    if not args.coco_face and lm_file.exists():
        lms, lm_valid = load_landmarks(lm_file, w, h)
        groups = {k: v for k, v in groups.items() if k != "face"}
        print(f"face: {lm_file.name}")
    out_path = args.output or args.video.with_name(args.video.stem + "_kpts.mp4")
    writer = None
    if args.output or not args.show:
        # pipe raw frames to ffmpeg for H.264 (much smaller than OpenCV's mp4v)
        writer = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out_path)],
            stdin=subprocess.PIPE,
        )

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(kpts):
            break
        if lms is not None and i < len(lms) and lm_valid[i]:
            draw_landmarks(frame, lms[i], GROUPS["face"][1], 1)
        if valid[i]:
            x0, y0, x1, y1 = boxes[i].astype(int)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)
            for idx, color in groups.values():
                for x, y, s in kpts[i, list(idx)]:
                    if s >= args.min_score:
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        cv2.putText(frame, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if writer:
            writer.stdin.write(frame.tobytes())
        if args.show:
            cv2.imshow("keypoints", cv2.resize(frame, (w // 2, h // 2)))
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break
        i += 1

    cap.release()
    if writer:
        writer.stdin.close()
        writer.wait()
        print(f"wrote {out_path} ({i} frames)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
