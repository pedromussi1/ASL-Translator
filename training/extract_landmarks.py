"""Extract MediaPipe hand landmarks from a folder of ASL alphabet images.

Expected input layout (e.g. Kaggle's "ASL Alphabet" dataset):
    data/asl_alphabet_train/A/A1.jpg
    data/asl_alphabet_train/A/A2.jpg
    ...
    data/asl_alphabet_train/Z/...

We walk every class folder, run MediaPipe HandLandmarker on each image,
normalize the landmarks the same way the browser does (wrist-relative,
scale-invariant, mirror left hands), and serialize one big NPZ that the
training script consumes.

Output:
    data/landmarks.npz
        X: float32 [N, 63]
        y: int64   [N]   (class index)
        labels: list[str]   (label vocabulary, in index order)

Run from the `training/` directory:
    python extract_landmarks.py --input data/asl_alphabet_train --output data/landmarks.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

CLASS_VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")


def normalize_landmarks(pts: np.ndarray, handedness: str, mirror_left: bool = True) -> np.ndarray:
    """Mirror of `src/lib/recognition/normalize.ts`. Keeps train and inference aligned.

    pts: [21, 3] float32, x/y normalized to [0,1] and z relative to wrist.
    """
    wrist = pts[0]
    middle_mcp = pts[9]
    delta = middle_mcp - wrist
    scale = float(np.linalg.norm(delta)) or 1e-6

    out = (pts - wrist) / scale
    if mirror_left and handedness == "Left":
        out[:, 0] *= -1
    return out.astype(np.float32).reshape(-1)


def discover_classes(root: Path) -> list[str]:
    found = sorted(p.name for p in root.iterdir() if p.is_dir())
    # Keep the canonical order; warn if anything is missing.
    missing = set(CLASS_VOCAB) - set(found)
    extra = set(found) - set(CLASS_VOCAB)
    if missing:
        print(f"WARN: missing class folders: {sorted(missing)}", file=sys.stderr)
    if extra:
        print(f"WARN: ignoring unexpected folders: {sorted(extra)}", file=sys.stderr)
    return [c for c in CLASS_VOCAB if c in found]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder of class subdirs containing images.")
    ap.add_argument("--output", required=True, help="Path to write the NPZ.")
    ap.add_argument("--max-per-class", type=int, default=None, help="Cap samples per class (debug).")
    ap.add_argument("--no-mirror", action="store_true", help="Disable left-hand mirroring.")
    args = ap.parse_args()

    in_root = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(in_root)
    if not classes:
        print(f"No class folders found under {in_root}", file=sys.stderr)
        return 2

    landmarker = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    skipped = 0

    for cls_idx, cls_name in enumerate(classes):
        files = sorted((in_root / cls_name).glob("*"))
        if args.max_per_class:
            files = files[: args.max_per_class]
        for fpath in tqdm(files, desc=f"{cls_name}", leave=False):
            img = cv2.imread(str(fpath))
            if img is None:
                skipped += 1
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = landmarker.process(rgb)
            if not res.multi_hand_landmarks or not res.multi_handedness:
                skipped += 1
                continue
            lm = res.multi_hand_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            handed = res.multi_handedness[0].classification[0].label  # "Left"/"Right"
            vec = normalize_landmarks(pts, handed, mirror_left=not args.no_mirror)
            X_rows.append(vec)
            y_rows.append(cls_idx)

    landmarker.close()

    if not X_rows:
        print("No usable samples extracted.", file=sys.stderr)
        return 3

    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    np.savez_compressed(out_path, X=X, y=y, labels=np.array(classes, dtype=object))
    print(f"Wrote {out_path}: {X.shape} samples, {len(classes)} classes, skipped {skipped}.")
    print(f"Labels (in index order): {json.dumps(classes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
