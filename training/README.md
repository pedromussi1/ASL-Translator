# Training

Phase 1 trains a tiny static-pose MLP that maps a single frame of normalized hand landmarks (63 floats) to one of **26 classes (A–Z)**. The model is exported to ONNX and dropped into `public/models/asl_alphabet_v1/` where the browser picks it up.

> The pipeline can also handle digits (0–9) if you point it at a dataset that includes them — see "Class vocabulary" below. The bundled training run uses [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), which is letters only.

## TL;DR

```sh
# 1) Make a Python venv (from inside the training/ folder)
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# or:  .venv\Scripts\activate     # Windows PowerShell

# 2A) Just want a stub ONNX so the browser pipeline can be wired up?  Tiny install.
pip install "onnx>=1.16" "numpy>=1.26"
python synthetic_warmup.py

# 2B) Want to actually train on Kaggle ASL Alphabet?    ~3 GB install.
pip install -r requirements.txt

# Set up Kaggle auth (https://www.kaggle.com/settings → API → Create New Token).
# The new bearer-token format goes in ~/.kaggle/access_token (a single line of "KGAT_…").

kaggle datasets download grassknoted/asl-alphabet -p data --unzip
python extract_landmarks.py \
    --input data/asl_alphabet_train/asl_alphabet_train \
    --output data/landmarks.npz \
    --max-per-class 1000        # 1k/class is plenty for this single-signer dataset
python train_alphabet.py --landmarks data/landmarks.npz
```

The Kaggle archive expands with a doubly-nested `asl_alphabet_train/asl_alphabet_train/` — that's not a typo, that's how the dataset ships.

## Files

| File | Purpose |
|---|---|
| `synthetic_warmup.py` | Build a tiny **stub** ONNX with random weights. Useful for wiring the browser pipeline before real training is done. Predictions are meaningless — the IO contract is what matters. |
| `extract_landmarks.py` | Walk a folder of ASL alphabet images, run **MediaPipe Tasks `HandLandmarker`** on each (the legacy `mp.solutions.hands` API is gone in MediaPipe 0.10.20+), normalize, write a single NPZ. Auto-downloads the `hand_landmarker.task` model on first run into `training/.cache/`. |
| `train_alphabet.py` | Train a 2-layer MLP (63 → 128 → 64 → num_classes) on the NPZ. Per-batch online augmentation: random 3D rotation about the wrist (±15°), uniform scale jitter (±10%), per-landmark Gaussian noise (σ=0.01). Exports a single self-contained ONNX (`dynamo=False` so weights don't split into a `.data` sidecar). |
| `requirements.txt` | Pinned-loose dependency list. |

## Class vocabulary

`extract_landmarks.py` keeps any class folder it finds whose name matches A–Z or 0–9 (in `CLASS_VOCAB`), in canonical order. Folders like `del`, `space`, `nothing` (which the Kaggle alphabet dataset includes) are ignored with a warning.

If your dataset is missing some classes, training proceeds with whatever's present. The browser reads `labels.json` dynamically, so any class count works — it doesn't have to be 26 or 36.

## Augmentation

Online augmentation in `train_alphabet.py` exists to close the cross-signer gap on a single-signer dataset (see [ADR-008](../docs/DECISIONS.md)). The held-out accuracy regresses slightly with augmentation enabled (we measured 98.34% → 98.06%) — that's the trade we want.

## IO contract

The browser-side [`AlphabetClassifier`](../src/lib/recognition/classifier.ts) expects:

- Input tensor name `input`, shape `[1, 63]`, dtype `float32`. Values are landmarks normalized by [`src/lib/recognition/normalize.ts`](../src/lib/recognition/normalize.ts) — wrist-relative, scaled by wrist-to-middle-MCP distance, left hand mirrored. The Python `normalize_landmarks` in `extract_landmarks.py` mirrors this exactly.
- Output tensor name `output`, shape `[1, N]`, dtype `float32`. Raw logits — softmax happens in JS.
- A sibling `labels.json` listing N strings indexed by class.

Both `synthetic_warmup.py` and `train_alphabet.py` produce this contract exactly. Swapping the trained model for the stub is a file copy of `alphabet.onnx` and `labels.json`.
