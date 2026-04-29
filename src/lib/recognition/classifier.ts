import type { NormalizedHand, Prediction } from "./types";

/**
 * Browser ONNX classifier wrapper.
 *
 * Phase 1 contract:
 *   - Input: a Float32 tensor of shape [1, 63] (21 landmarks × 3 coords).
 *   - Output: raw logits of shape [1, num_classes]. We apply softmax here.
 *   - Labels are loaded from a JSON file: an array of strings indexed by class.
 *
 * The wrapper is intentionally tiny — it doesn't know about word buffers or
 * temporal smoothing. That's the consumer's job.
 */
export class AlphabetClassifier {
  private session: import("onnxruntime-web").InferenceSession | null = null;
  private labels: string[] = [];
  private inputName: string = "input";
  private outputName: string = "output";

  private constructor() {}

  static async load(opts: {
    modelUrl: string;
    labelsUrl: string;
  }): Promise<AlphabetClassifier> {
    if (typeof window === "undefined") {
      throw new Error("AlphabetClassifier is browser-only.");
    }
    const ort = await import("onnxruntime-web");

    // Use bundled WASM binaries from the package; ORT figures out the right
    // variant (single-threaded vs threaded vs SIMD) based on the browser.
    // See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
    const [session, labelsRes] = await Promise.all([
      ort.InferenceSession.create(opts.modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
      fetch(opts.labelsUrl).then((r) => {
        if (!r.ok) throw new Error(`labels fetch failed: ${r.status}`);
        return r.json() as Promise<string[]>;
      }),
    ]);

    const c = new AlphabetClassifier();
    c.session = session;
    c.labels = labelsRes;
    c.inputName = session.inputNames[0] ?? "input";
    c.outputName = session.outputNames[0] ?? "output";
    return c;
  }

  /**
   * Run inference on a normalized hand. Returns the top-1 label and its
   * softmaxed confidence in [0, 1].
   */
  async recognize(hand: NormalizedHand): Promise<Prediction> {
    if (!this.session) throw new Error("Classifier not loaded.");
    const ort = await import("onnxruntime-web");

    const tensor = new ort.Tensor("float32", hand.vector, [1, hand.vector.length]);
    const out = await this.session.run({ [this.inputName]: tensor });
    const logits = out[this.outputName].data as Float32Array;

    const { idx, prob } = topkSoftmax(logits);
    const label = this.labels[idx] ?? `cls_${idx}`;
    return { label, confidence: prob };
  }

  get classNames(): readonly string[] {
    return this.labels;
  }

  close(): void {
    this.session?.release?.();
    this.session = null;
  }
}

/** Argmax + softmax for the winning class only (numerically stable). */
function topkSoftmax(logits: Float32Array): { idx: number; prob: number } {
  let maxV = -Infinity;
  let maxIdx = 0;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxV) {
      maxV = logits[i];
      maxIdx = i;
    }
  }
  let denom = 0;
  for (let i = 0; i < logits.length; i++) denom += Math.exp(logits[i] - maxV);
  const prob = denom > 0 ? 1 / denom : 0;
  return { idx: maxIdx, prob };
}
