# Training

Phase 1 trains a tiny static-pose MLP that maps a single frame of normalized hand landmarks (63 floats) to one of 36 classes (A–Z + 0–9). The model is exported to ONNX and dropped into `public/models/asl_alphabet_v1/` where the browser picks it up.

## TL;DR

```sh
# 1) Make a Python venv
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# or:  .venv\Scripts\activate     # Windows PowerShell

# 2A) Just want a stub model so you can develop the browser side?  ~50 MB install.
pip install onnx>=1.16 numpy>=1.26
python synthetic_warmup.py

# 2B) Want to actually train on the Kaggle ASL Alphabet dataset?    ~3 GB install.
pip install -r requirements.txt
python extract_landmarks.py --input data/asl_alphabet_train --output data/landmarks.npz
python train_alphabet.py --landmarks data/landmarks.npz
```

## Files

| File | Purpose |
|---|---|
| `synthetic_warmup.py` | Build a tiny **stub** ONNX with random weights. Useful for wiring the browser pipeline before training is done. Predictions are meaningless — the IO contract is what matters. |
| `extract_landmarks.py` | Walk a folder of ASL alphabet images, run MediaPipe Hands on each, normalize, and write a single NPZ. |
| `train_alphabet.py` | Train a 2-layer MLP on the NPZ; report metrics and export to ONNX with the same IO contract as the stub. |
| `requirements.txt` | Pinned-loose dependency list. |

## Dataset

We expect a Kaggle "ASL Alphabet"-style folder layout:

```
data/asl_alphabet_train/
├── A/  (lots of images of hand making the A sign)
├── B/
├── ...
└── Z/
```

Recommended sources (CC-licensed variants suitable for personal use; **not commercial without checking each license**):

- [ASL Alphabet (Akash) — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [Sign Language MNIST — Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (lower quality but tiny)

Numbers (0–9) are usually a separate dataset; combine them under the same root or skip and limit to A–Z by editing `CLASS_VOCAB` in `extract_landmarks.py`.

## IO contract

The browser-side `AlphabetClassifier` expects:

- Input tensor name `input`, shape `[1, 63]`, dtype `float32`. Values are landmarks normalized by `src/lib/recognition/normalize.ts` (wrist-relative, scaled by wrist-to-middle-MCP distance, left hand mirrored).
- Output tensor name `output`, shape `[1, N]`, dtype `float32`. Raw logits — softmax happens in JS.
- A sibling `labels.json` listing N strings indexed by class.

Both `synthetic_warmup.py` and `train_alphabet.py` produce this contract exactly. Swapping the trained model for the stub is a file copy.
