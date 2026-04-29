import type { NormalizedHand, Prediction } from "./types";

/** Frames per class to collect during a calibration session. */
export const CALIBRATION_FRAMES_PER_LETTER = 60;

/** localStorage key for the persisted calibration. */
export const CALIBRATION_STORAGE_KEY = "asl-translator:calibration:v1";

export interface CalibrationData {
  version: 1;
  createdAt: number;
  prototypes: Record<string, number[]>; // label → 63-d mean vector
  sampleCounts: Record<string, number>;
}

/**
 * Mutable session that collects frames letter-by-letter, then exports a
 * `CalibrationData` record. The calling UI drives the letter sequence.
 */
export class CalibrationSession {
  private buffers = new Map<string, Float32Array[]>();

  constructor(public readonly letters: readonly string[]) {}

  recordedCount(letter: string): number {
    return this.buffers.get(letter)?.length ?? 0;
  }

  push(letter: string, hand: NormalizedHand): void {
    let arr = this.buffers.get(letter);
    if (!arr) {
      arr = [];
      this.buffers.set(letter, arr);
    }
    arr.push(hand.vector);
  }

  /** Drop everything collected for this letter — used by a "Retry" button. */
  reset(letter: string): void {
    this.buffers.delete(letter);
  }

  /** Letters that have at least one recorded frame. */
  recordedLetters(): string[] {
    return [...this.buffers.keys()].filter((k) => (this.buffers.get(k)?.length ?? 0) > 0);
  }

  /** Build the persistable artifact. Letters with zero frames are excluded. */
  build(): CalibrationData {
    const prototypes: Record<string, number[]> = {};
    const sampleCounts: Record<string, number> = {};
    for (const [letter, frames] of this.buffers) {
      if (frames.length === 0) continue;
      const mean = new Float32Array(63);
      for (const v of frames) {
        for (let i = 0; i < 63; i++) mean[i] += v[i];
      }
      for (let i = 0; i < 63; i++) mean[i] /= frames.length;
      prototypes[letter] = Array.from(mean);
      sampleCounts[letter] = frames.length;
    }
    return {
      version: 1,
      createdAt: Date.now(),
      prototypes,
      sampleCounts,
    };
  }
}

/**
 * Per-user prototype classifier. Built once from calibration data and reused
 * for every frame. Predicts via L2 to mean prototypes; confidence is the
 * margin between nearest and second-nearest matches (SIFT-style ratio).
 */
export class CalibratedClassifier {
  private readonly labels: string[];
  private readonly prototypes: Float32Array[]; // parallel to labels

  private constructor(labels: string[], prototypes: Float32Array[]) {
    this.labels = labels;
    this.prototypes = prototypes;
  }

  static fromData(data: CalibrationData): CalibratedClassifier {
    const labels: string[] = [];
    const prototypes: Float32Array[] = [];
    for (const [letter, vec] of Object.entries(data.prototypes)) {
      labels.push(letter);
      prototypes.push(Float32Array.from(vec));
    }
    return new CalibratedClassifier(labels, prototypes);
  }

  static loadFromStorage(): CalibratedClassifier | null {
    if (typeof localStorage === "undefined") return null;
    const raw = localStorage.getItem(CALIBRATION_STORAGE_KEY);
    if (!raw) return null;
    try {
      const data = JSON.parse(raw) as CalibrationData;
      if (data.version !== 1 || !data.prototypes) return null;
      if (Object.keys(data.prototypes).length === 0) return null;
      return CalibratedClassifier.fromData(data);
    } catch {
      return null;
    }
  }

  static saveToStorage(data: CalibrationData): void {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(data));
  }

  static clearStorage(): void {
    if (typeof localStorage === "undefined") return;
    localStorage.removeItem(CALIBRATION_STORAGE_KEY);
  }

  async recognize(hand: NormalizedHand): Promise<Prediction> {
    return this.recognizeSync(hand);
  }

  /** Synchronous variant — useful for tests / non-async call sites. */
  recognizeSync(hand: NormalizedHand): Prediction {
    if (this.prototypes.length === 0) {
      return { label: "?", confidence: 0 };
    }
    const v = hand.vector;
    let nearest = 0;
    let nearestSq = sqDist(v, this.prototypes[0]);
    let secondSq = Number.POSITIVE_INFINITY;
    for (let i = 1; i < this.prototypes.length; i++) {
      const d = sqDist(v, this.prototypes[i]);
      if (d < nearestSq) {
        secondSq = nearestSq;
        nearestSq = d;
        nearest = i;
      } else if (d < secondSq) {
        secondSq = d;
      }
    }
    // Margin-based confidence: 1 - sqrt(nearest)/sqrt(second).
    // Equivalent in ranking to SIFT's ratio test, bounded in [0, 1).
    const ratio =
      secondSq === Number.POSITIVE_INFINITY
        ? 0
        : Math.sqrt(nearestSq) / Math.max(1e-6, Math.sqrt(secondSq));
    const confidence = Math.max(0, Math.min(1, 1 - ratio));
    return { label: this.labels[nearest], confidence };
  }

  get classNames(): readonly string[] {
    return this.labels;
  }

  close(): void {
    /* no-op — kept for parity with AlphabetClassifier */
  }
}

function sqDist(a: Float32Array, b: Float32Array): number {
  let s = 0;
  // Both vectors are length 63; unrolled per pair for the hot loop.
  for (let i = 0; i < 63; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}
