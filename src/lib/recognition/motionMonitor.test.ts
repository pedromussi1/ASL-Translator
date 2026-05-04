import { describe, expect, it } from "vitest";
import { MotionMonitor } from "./motionMonitor";
import type { DetectedHand, Point3 } from "./types";

/**
 * Build a 21-landmark hand where every landmark is at the same point. This is
 * unrealistic but keeps the tests focused — only the fingertip displacements
 * (indices 4, 8, 12, 16, 20) are read by MotionMonitor, and we override those
 * directly.
 */
function mkHand(opts: {
  pinkyTip?: { x: number; y: number; z?: number };
  indexTip?: { x: number; y: number; z?: number };
}): DetectedHand {
  const lms: Point3[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  if (opts.pinkyTip) lms[20] = { x: opts.pinkyTip.x, y: opts.pinkyTip.y, z: opts.pinkyTip.z ?? 0 };
  if (opts.indexTip) lms[8] = { x: opts.indexTip.x, y: opts.indexTip.y, z: opts.indexTip.z ?? 0 };
  return {
    landmarks: lms,
    worldLandmarks: lms,
    handedness: "Right",
    handednessScore: 0.95,
  };
}

describe("MotionMonitor", () => {
  it("reports no motion for a perfectly still hand", () => {
    const m = new MotionMonitor();
    for (let i = 0; i < 10; i++) {
      const moving = m.push(mkHand({ pinkyTip: { x: 0.5, y: 0.5 } }));
      expect(moving).toBe(false);
    }
  });

  it("flags motion when one fingertip moves fast across frames", () => {
    const m = new MotionMonitor();
    // Pinky tip moves 0.01 per frame; over 5 frames that's 0.05, well above
    // the default 0.025 threshold.
    let moving = false;
    for (let i = 0; i < 6; i++) {
      moving = m.push(mkHand({ pinkyTip: { x: 0.4 + i * 0.01, y: 0.4 } }));
    }
    expect(moving).toBe(true);
  });

  it("does NOT flag motion for the natural drift of a held pose", () => {
    const m = new MotionMonitor();
    // Sub-threshold jitter at 0.001/frame ⇒ window total ~0.005 ≪ 0.025.
    let moving = false;
    for (let i = 0; i < 10; i++) {
      moving = m.push(mkHand({ pinkyTip: { x: 0.5 + Math.sin(i) * 0.001, y: 0.5 } }));
    }
    expect(moving).toBe(false);
  });

  it("captures fast motion of a single finger even when others are still", () => {
    const m = new MotionMonitor();
    // Index tip oscillates fast while pinky stays still — Z-style motion.
    let moving = false;
    for (let i = 0; i < 6; i++) {
      moving = m.push(
        mkHand({
          indexTip: { x: 0.4 + i * 0.012, y: 0.4 },
          pinkyTip: { x: 0.5, y: 0.5 },
        }),
      );
    }
    expect(moving).toBe(true);
  });

  it("resets when no hand is visible (no false carryover)", () => {
    const m = new MotionMonitor();
    // Build up some motion …
    for (let i = 0; i < 5; i++) {
      m.push(mkHand({ pinkyTip: { x: 0.4 + i * 0.02, y: 0.4 } }));
    }
    // … then lose the hand …
    expect(m.push(null)).toBe(false);
    // … and the next pose should NOT immediately read as in-motion.
    expect(m.push(mkHand({ pinkyTip: { x: 0.5, y: 0.5 } }))).toBe(false);
  });
});
