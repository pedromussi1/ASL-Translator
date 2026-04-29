"""Train the static-letter MLP on extracted landmarks and export to ONNX.

Pipeline:
    1. `extract_landmarks.py` → data/landmarks.npz
    2. THIS SCRIPT          → ../public/models/asl_alphabet_v1/{alphabet.onnx, labels.json}

The exported ONNX has the same IO contract as `synthetic_warmup.py`:
    input  "input"  float32 [1, 63]   (21 landmarks * 3 coords, normalized)
    output "output" float32 [1, num_classes]   (raw logits — softmax in JS)

Run from the `training/` directory:
    python train_alphabet.py --landmarks data/landmarks.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "public" / "models" / "asl_alphabet_v1"


class AlphabetMLP(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = 63, hidden=(128, 64)):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", required=True, help="Path to landmarks.npz from extract_landmarks.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0xA5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz = np.load(args.landmarks, allow_pickle=True)
    X = npz["X"].astype(np.float32)
    y = npz["y"].astype(np.int64)
    labels: list[str] = list(npz["labels"])
    num_classes = len(labels)
    print(f"Loaded {X.shape[0]} samples, {num_classes} classes.")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AlphabetMLP(num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    Xte_t = torch.from_numpy(Xte).to(device)
    yte_t = torch.from_numpy(yte).to(device)

    for epoch in range(args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * xb.size(0)
            n += xb.size(0)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            te_logits = model(Xte_t)
            te_acc = (te_logits.argmax(1) == yte_t).float().mean().item()
        print(f"epoch {epoch+1:02d}  train_loss={loss_sum/n:.4f}  test_acc={te_acc*100:.2f}%")

    model.eval()
    with torch.no_grad():
        te_logits = model(Xte_t).cpu().numpy()
    te_pred = te_logits.argmax(1)
    print("\nFinal classification report:")
    print(classification_report(yte, te_pred, target_names=labels, digits=3))
    cm = confusion_matrix(yte, te_pred)
    np.set_printoptions(linewidth=200)
    print("Confusion matrix (rows = truth, cols = pred):")
    print(cm)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "alphabet.onnx"
    labels_path = OUTPUT_DIR / "labels.json"

    dummy = torch.zeros(1, 63, device=device)
    # Use the legacy TorchScript exporter (`dynamo=False`) so we get a single
    # self-contained .onnx file without an external `.data` sidecar. Our model
    # is ~70 KB — external data only complicates static serving for no benefit.
    torch.onnx.export(
        model,
        dummy,
        str(model_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,  # batch size fixed to 1 — fine for live inference
        dynamo=False,
    )
    labels_path.write_text(json.dumps(labels, indent=2))

    size_kb = model_path.stat().st_size / 1024
    print(f"\nWrote {model_path} ({size_kb:.1f} KB)")
    print(f"Wrote {labels_path} ({num_classes} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
