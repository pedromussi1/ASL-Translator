import type { DetectedHand, Point3 } from "./types";

const FINGERTIP_INDICES = [4, 8, 12, 16, 20] as const;
const DEFAULT_WINDOW = 5;
/**
 * Sum-of-max-fingertip-speed threshold over the window. In normalized image
 * coords (landmarks are in [0, 1]), ~0.001–0.003/frame is a steady hand,
 * ~0.005+/frame is meaningful motion. With a 5-frame window, 0.025 reliably
 * separates fingerspelling-style holds from dynamic-sign motion.
 */
const DEFAULT_THRESHOLD = 0.025;

/**
 * Tracks whether the user's hand is currently moving "dynamically" — useful
 * to suppress static-letter commits during dynamic signs (e.g. so "I" doesn't
 * get committed before "J" finishes drawing).
 *
 * Per frame, picks the fastest-moving fingertip vs the previous frame and
 * sums those max-speeds across a rolling window. If the window total exceeds
 * `threshold`, the user is in motion. Tracking the *max* (not the average)
 * matters because dynamic signs like J move only one finger meaningfully —
 * averaging across fingers would dilute the signal.
 */
export class MotionMonitor {
  private prevTips: Point3[] | null = null;
  private buf: number[] = [];

  constructor(
    private readonly windowSize: number = DEFAULT_WINDOW,
    private readonly threshold: number = DEFAULT_THRESHOLD,
  ) {}

  /** Push the latest detected hand. Returns true if the user is in motion. */
  push(hand: DetectedHand | null): boolean {
    if (!hand) {
      this.reset();
      return false;
    }
    const tips = FINGERTIP_INDICES.map((i) => hand.landmarks[i]);
    if (this.prevTips) {
      let max = 0;
      for (let i = 0; i < tips.length; i++) {
        const d = dist3(tips[i], this.prevTips[i]);
        if (d > max) max = d;
      }
      this.buf.push(max);
      if (this.buf.length > this.windowSize) this.buf.shift();
    }
    this.prevTips = tips;
    return this.totalMotion() > this.threshold;
  }

  totalMotion(): number {
    let s = 0;
    for (const x of this.buf) s += x;
    return s;
  }

  reset(): void {
    this.prevTips = null;
    this.buf = [];
  }
}

function dist3(a: Point3, b: Point3): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
