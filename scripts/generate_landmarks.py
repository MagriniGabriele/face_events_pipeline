"""Dense face landmarks for a face crop -> <stem>_landmarks.npz (annotations only, no video).

Two methods:
  mediapipe  (default) MediaPipe Face Landmarker: 478 points per frame, tracked across
             frames (VIDEO mode), plus 52 blendshapes and the 4x4 head pose matrix.
             Needs landmark_models/face_landmarker.task (setup.sh downloads it). CPU by
             default: --device gpu (OpenGL ES via EGL, not CUDA) is ~1.7x faster for one
             clip (216 vs 126 fps on an RTX 4070), but over a dataset 8 CPU workers beat
             it (~800 vs ~500 fps in total) and leave the GPU to eventcv.
  fa         face_alignment (FAN): the 68-point iBUG layout, detected independently in
             every frame; needs face_alignment + torch (setup.sh), GPU (CUDA) if present.

Both give far more precise face points than the dataset's COCO-WholeBody face keypoints,
which sit on a coarse ~15 px grid. The other scripts (visualize_keypoints, render_events,
overlay_events) draw these landmarks instead of the COCO face keypoints when the file exists.

Output `<stem>_landmarks.npz`, one row per video frame (row i = frame i):
  landmarks      (T, N, 2) float32  x, y in pixels of the video; NaN where no face
  valid          (T,)      bool     a face was found in frame i
  width, height, fps, method        video size the pixels refer to; "mediapipe" or "fa"
  mediapipe only:
  z              (T, N)    float32  relative depth, same scale as x (smaller = closer)
  blendshapes    (T, 52)   float32  blendshape scores; names in `blendshape_names`
  transform      (T, 4, 4) float32  facial transformation matrix (head pose)

Usage:
  python scripts/generate_landmarks.py OUT/.../<id>_face640.mp4 [--method fa] [--max-frames N]
  (for every clip of a dataset folder, see scripts/landmarks_all.py)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path(os.environ.get("LANDMARK_MODEL", REPO / "landmark_models" / "face_landmarker.task"))
METHODS = ("mediapipe", "fa")


def landmarks_path(clip: Path) -> Path:
    """<stem>_landmarks.npz for any file of a clip (<id>_face640.mp4/.npz/_events.h5)."""
    stem = clip.name.removesuffix("_events.h5").removesuffix(".mp4").removesuffix(".npz")
    return clip.with_name(f"{stem}_landmarks.npz")


def load_landmarks(path: Path, width: int | None = None, height: int | None = None):
    """(landmarks (T, N, 2), valid (T,)), rescaled to width x height (e.g. EVENTS_SCALE)."""
    data = np.load(path)
    pts, valid = data["landmarks"].copy(), data["valid"]
    if width and height:
        pts[..., 0] *= width / int(data["width"])
        pts[..., 1] *= height / int(data["height"])
    return pts, valid


def draw_landmarks(img, pts, color, radius: int = 1) -> None:
    import cv2

    for x, y in pts[~np.isnan(pts).any(1)]:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1)


def read_frames(video: Path, max_frames: int | None):
    """Yield RGB frames; also returns fps, width, height through the first item."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    yield fps, w, h
    i = 0
    try:
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def run_mediapipe(frames, fps: float, w: int, h: int, model: Path, device: str = "cpu") -> dict:
    os.environ.setdefault("GLOG_minloglevel", "2")  # before the import: silences mediapipe's C++ log noise
    import mediapipe as mp

    if not model.exists():
        raise FileNotFoundError(f"{model} not found (run setup.sh, or pass --model)")
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model),
            delegate=mp.tasks.BaseOptions.Delegate.GPU if device == "gpu" else mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,  # tracks across frames: steadier and faster
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    pts, z, valid, blend, transform, names = [], [], [], [], [], None
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, rgb in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # timestamps must strictly increase: derive them from the frame index
            res = landmarker.detect_for_video(image, int(round(i * 1000 / fps)))
            if res.face_landmarks:
                lm = np.array([(p.x * w, p.y * h, p.z * w) for p in res.face_landmarks[0]], np.float32)
                pts.append(lm[:, :2])
                z.append(lm[:, 2])
                valid.append(True)
                blend.append([c.score for c in res.face_blendshapes[0]])
                names = names or [c.category_name for c in res.face_blendshapes[0]]
                transform.append(np.asarray(res.facial_transformation_matrixes[0], np.float32))
            else:
                pts.append(None); z.append(None); blend.append(None); transform.append(None)
                valid.append(False)
    n_pts = next((p.shape[0] for p in pts if p is not None), 478)
    n_blend = len(names) if names else 52

    def stack(rows, shape):
        nan = np.full(shape, np.nan, np.float32)
        return np.stack([nan if r is None else np.asarray(r, np.float32) for r in rows]) if rows \
            else np.zeros((0, *shape), np.float32)

    return {"landmarks": stack(pts, (n_pts, 2)), "valid": np.array(valid, bool),
            "z": stack(z, (n_pts,)), "blendshapes": stack(blend, (n_blend,)),
            "blendshape_names": np.array(names or []), "transform": stack(transform, (4, 4))}


def run_fa(frames, device: str | None) -> dict:
    try:
        import face_alignment
        import torch
    except ImportError as e:
        raise ImportError("--method fa needs face_alignment: pip install face-alignment") from e

    device = "cpu" if device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device,
                                      flip_input=False, compile=False)
    pts, valid = [], []
    for rgb in frames:
        faces = fa.get_landmarks_from_image(rgb)
        if faces:
            # several detections: keep the largest face (the crop is centred on the subject)
            face = max(faces, key=lambda f: np.ptp(f[:, 0]) * np.ptp(f[:, 1]))
            pts.append(face[:, :2].astype(np.float32)); valid.append(True)
        else:
            pts.append(np.full((68, 2), np.nan, np.float32)); valid.append(False)
    return {"landmarks": np.stack(pts) if pts else np.zeros((0, 68, 2), np.float32),
            "valid": np.array(valid, bool)}


def generate(video: Path, method: str = "mediapipe", model: Path = DEFAULT_MODEL,
             out: Path | None = None, max_frames: int | None = None,
             device: str | None = None) -> Path:
    """Landmarks for every frame of `video` -> out (default <stem>_landmarks.npz)."""
    out = out or landmarks_path(video)
    frames = read_frames(video, max_frames)
    fps, w, h = next(frames)
    t0 = time.time()
    if method == "mediapipe":
        data = run_mediapipe(frames, fps, w, h, Path(model), device or "cpu")
    elif method == "fa":
        data = run_fa(frames, device)
    else:
        raise ValueError(f"unknown method {method!r} (choose from {METHODS})")
    if not len(data["valid"]):
        raise RuntimeError(f"no frames decoded from {video}")
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, width=w, height=h, fps=fps, method=method, **data)
    os.replace(tmp, out)  # atomic: a partial file is never mistaken for a finished one
    n, found = len(data["valid"]), int(data["valid"].sum())
    print(f"  landmarks {out.name}: {method}, {found}/{n} frames with a face, "
          f"{n / max(time.time() - t0, 1e-6):.0f} fps", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path, nargs="+", help="<id>_face640.mp4 (one or more)")
    ap.add_argument("--method", choices=METHODS, default="mediapipe")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="mediapipe .task model (env LANDMARK_MODEL)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default=os.environ.get("LANDMARKS_DEVICE"),
                    help="default: cpu for mediapipe, gpu (if available) for fa (env LANDMARKS_DEVICE)")
    ap.add_argument("-o", "--output", type=Path, help="output .npz (only with a single video)")
    ap.add_argument("--max-frames", type=int, help="testing only: first N frames")
    args = ap.parse_args()
    if args.output and len(args.video) > 1:
        ap.error("-o needs a single video")
    for video in args.video:
        print(video.name, flush=True)
        generate(video, args.method, args.model, args.output, args.max_frames, args.device)


if __name__ == "__main__":
    main()
