import type { Prediction } from "./types";

/**
 * Sliding-window prediction smoother. Sits between the per-frame classifier
 * and the WordBuffer to drop transient flicker — a single-frame mis-prediction
 * for "T" while you're holding "A" shouldn't escape into the buffer.
 *
 * Strategy: keep the last N frame results. The smoothed output is whichever
 * label (or null) appears most often in the window. Confidence is averaged
 * over agreeing frames.
 */
export class PredictionSmoother {
  private buf: Array<Prediction | null> = [];

  constructor(private readonly windowSize: number = 5) {}

  push(p: Prediction | null): Prediction | null {
    this.buf.push(p);
    if (this.buf.length > this.windowSize) this.buf.shift();
    if (this.buf.length === 0) return null;

    type Bucket = { count: number; sumConf: number };
    const counts = new Map<string | null, Bucket>();
    for (const item of this.buf) {
      const key = item ? item.label : null;
      const b = counts.get(key) ?? { count: 0, sumConf: 0 };
      b.count++;
      b.sumConf += item?.confidence ?? 0;
      counts.set(key, b);
    }

    let bestKey: string | null = null;
    let bestCount = -1;
    let bestSum = 0;
    for (const [key, { count, sumConf }] of counts) {
      // Tie-break by total confidence so an exact tie picks the more confident label.
      if (count > bestCount || (count === bestCount && sumConf > bestSum)) {
        bestKey = key;
        bestCount = count;
        bestSum = sumConf;
      }
    }

    if (bestKey === null) return null;
    return { label: bestKey, confidence: bestSum / bestCount };
  }

  reset(): void {
    this.buf = [];
  }
}
