"""Build a tiny stub ONNX classifier so the browser pipeline can be wired
end-to-end before the real model is trained.

The stub is a 2-layer MLP (63 -> 128 -> 64 -> num_classes) with random
weights. Its predictions are meaningless — the point is only that it loads
in `onnxruntime-web`, accepts a `[1, 63]` Float32 input, and emits logits
over the configured label set. Real training (`train_alphabet.py`) produces
a model with the *same* IO contract, so swapping it in is a file copy.

Outputs:
  ../public/models/asl_alphabet_v1/alphabet.onnx
  ../public/models/asl_alphabet_v1/labels.json

Run from the `training/` directory:
  python synthetic_warmup.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LABELS: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    *list("0123456789"),
]
NUM_CLASSES = len(LABELS)
INPUT_DIM = 63
HIDDEN1 = 128
HIDDEN2 = 64

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "public" / "models" / "asl_alphabet_v1"


def make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array.astype(np.float32), name=name)


def build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(seed=0xA5)
    # Initializers (Gemm uses Y = A * B + C with B transposed by default,
    # so weights are stored in [out_dim, in_dim] orientation when transB=1).
    w1 = rng.standard_normal((HIDDEN1, INPUT_DIM)) * 0.1
    b1 = np.zeros(HIDDEN1)
    w2 = rng.standard_normal((HIDDEN2, HIDDEN1)) * 0.1
    b2 = np.zeros(HIDDEN2)
    w3 = rng.standard_normal((NUM_CLASSES, HIDDEN2)) * 0.1
    b3 = np.zeros(NUM_CLASSES)

    initializers = [
        make_initializer("W1", w1),
        make_initializer("b1", b1),
        make_initializer("W2", w2),
        make_initializer("b2", b2),
        make_initializer("W3", w3),
        make_initializer("b3", b3),
    ]

    input_v = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, INPUT_DIM])
    output_v = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, NUM_CLASSES])

    nodes = [
        helper.make_node("Gemm", ["input", "W1", "b1"], ["h1"], transB=1),
        helper.make_node("Relu", ["h1"], ["h1r"]),
        helper.make_node("Gemm", ["h1r", "W2", "b2"], ["h2"], transB=1),
        helper.make_node("Relu", ["h2"], ["h2r"]),
        helper.make_node("Gemm", ["h2r", "W3", "b3"], ["output"], transB=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="asl_alphabet_stub",
        inputs=[input_v],
        outputs=[output_v],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="asl-translator-synthetic-warmup",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9  # compatible with onnxruntime-web ~1.18+
    onnx.checker.check_model(model)
    return model


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model()
    model_path = OUTPUT_DIR / "alphabet.onnx"
    labels_path = OUTPUT_DIR / "labels.json"

    onnx.save(model, str(model_path))
    labels_path.write_text(json.dumps(LABELS, indent=2))

    size_kb = model_path.stat().st_size / 1024
    print(f"Wrote {model_path} ({size_kb:.1f} KB)")
    print(f"Wrote {labels_path} ({len(LABELS)} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
