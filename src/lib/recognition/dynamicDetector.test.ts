import { describe, expect, it } from "vitest";
import { DynamicSignDetector } from "./dynamicDetector";
import type { DetectedHand, Point3 } from "./types";

type Shape = "pinkyOut" | "indexOut" | "fist" | "open";
type Handedness = "Left" | "Right";

interface FrameSpec {
  shape: Shape;
  wrist?: { x: number; y: number };
  /** Override the pinky tip's x/y (for trajectory simulation). */
  pinkyTip?: { x: number; y: number };
  /** Override the index tip's x/y. */
  indexTip?: { x: number; y: number };
  handedness?: Handedness;
}

/**
 * Synthesize a 21-landmark MediaPipe-style hand. For extended fingers, PIP
 * and DIP are placed colinearly between MCP and TIP so the extension ratio
 * stays ≈1.0 even when we override the tip far from its natural position
 * (real MediaPipe always bends bones coherently with the tip).
 */
function mkFrame(spec: FrameSpec): DetectedHand {
  const wrist: Point3 = {
    x: spec.wrist?.x ?? 0.5,
    y: spec.wrist?.y ?? 0.5,
    z: 0,
  };
  const lms: Point3[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  lms[0] = wrist;

  const ext = {
    thumb: spec.shape === "open",
    index: spec.shape === "indexOut" || spec.shape === "open",
    middle: spec.shape === "open",
    ring: spec.shape === "open",
    pinky: spec.shape === "pinkyOut" || spec.shape === "open",
  };

  const placeFinger = (
    mcpStart: number,
    mcpPos: { x: number; y: number },
    isExtended: boolean,
    customTip?: { x: number; y: number },
  ) => {
    if (isExtended) {
      const tip = customTip ?? { x: mcpPos.x, y: mcpPos.y - 0.09 };
      const dx = tip.x - mcpPos.x;
      const dy = tip.y - mcpPos.y;
      lms[mcpStart] = { x: mcpPos.x, y: mcpPos.y, z: 0 };
      lms[mcpStart + 1] = { x: mcpPos.x + dx / 3, y: mcpPos.y + dy / 3, z: 0 };
      lms[mcpStart + 2] = { x: mcpPos.x + (2 * dx) / 3, y: mcpPos.y + (2 * dy) / 3, z: 0 };
      lms[mcpStart + 3] = { x: tip.x, y: tip.y, z: 0 };
    } else {
      lms[mcpStart] = { x: mcpPos.x, y: mcpPos.y, z: 0 };
      lms[mcpStart + 1] = { x: mcpPos.x, y: mcpPos.y - 0.03, z: 0 };
      lms[mcpStart + 2] = { x: mcpPos.x, y: mcpPos.y - 0.06, z: 0 };
      lms[mcpStart + 3] = customTip
        ? { x: customTip.x, y: customTip.y, z: 0 }
        : { x: mcpPos.x, y: mcpPos.y - 0.03, z: 0 };
    }
  };

  placeFinger(1, { x: wrist.x - 0.04, y: wrist.y - 0.02 }, ext.thumb);
  placeFinger(5, { x: wrist.x - 0.02, y: wrist.y - 0.05 }, ext.index, spec.indexTip);
  placeFinger(9, { x: wrist.x, y: wrist.y - 0.06 }, ext.middle);
  placeFinger(13, { x: wrist.x + 0.02, y: wrist.y - 0.05 }, ext.ring);
  placeFinger(17, { x: wrist.x + 0.04, y: wrist.y - 0.04 }, ext.pinky, spec.pinkyTip);

  return {
    landmarks: lms,
    worldLandmarks: lms,
    handedness: spec.handedness ?? "Right",
    handednessScore: 0.95,
  };
}

/** Run the detector through a sequence of frames at 30fps, return any fire. */
function simulate(
  detector: DynamicSignDetector,
  frames: DetectedHand[],
  startMs = 0,
): { label: string; confidence: number } | null {
  let last = null;
  let t = startMs;
  for (const f of frames) {
    const r = detector.push(f, t);
    if (r) last = r;
    t += 33; // ~30fps
  }
  return last;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// ---------------------------------------------------------------------------
// J — pinky out, traces J: down then curve toward thumb side
// ---------------------------------------------------------------------------

describe("DynamicSignDetector — J", () => {
  it("fires on a clear J motion (right hand)", () => {
    const N = 25;
    const frames: DetectedHand[] = [];
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      let x: number;
      let y: number;
      if (t < 0.6) {
        x = 0.55;
        y = lerp(0.3, 0.55, t / 0.6);
      } else {
        const tt = (t - 0.6) / 0.4;
        x = lerp(0.55, 0.45, tt);
        y = lerp(0.55, 0.62, tt);
      }
      frames.push(mkFrame({ shape: "pinkyOut", pinkyTip: { x, y }, handedness: "Right" }));
    }
    const r = simulate(new DynamicSignDetector(), frames);
    expect(r?.label).toBe("J");
    expect(r?.confidence).toBeGreaterThan(0.7);
  });

  it("does not fire when the hand is held still in pinky-out shape", () => {
    const frames = Array.from({ length: 30 }, () =>
      mkFrame({ shape: "pinkyOut", pinkyTip: { x: 0.5, y: 0.4 }, handedness: "Right" }),
    );
    const r = simulate(new DynamicSignDetector(), frames);
    expect(r).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Z — index out, traces three segments: right, diag-down-left, right
// ---------------------------------------------------------------------------

describe("DynamicSignDetector — Z", () => {
  it("fires on a clear Z motion (right hand)", () => {
    const N = 30;
    const frames: DetectedHand[] = [];
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      let x: number;
      let y: number;
      if (t < 1 / 3) {
        const tt = t / (1 / 3);
        x = lerp(0.55, 0.45, tt);
        y = 0.35;
      } else if (t < 2 / 3) {
        const tt = (t - 1 / 3) / (1 / 3);
        x = lerp(0.45, 0.55, tt);
        y = lerp(0.35, 0.5, tt);
      } else {
        const tt = (t - 2 / 3) / (1 / 3);
        x = lerp(0.55, 0.45, tt);
        y = 0.5;
      }
      frames.push(mkFrame({ shape: "indexOut", indexTip: { x, y }, handedness: "Right" }));
    }
    const r = simulate(new DynamicSignDetector(), frames);
    expect(r?.label).toBe("Z");
    expect(r?.confidence).toBeGreaterThan(0.7);
  });

  it("does not fire when the index is held still", () => {
    const frames = Array.from({ length: 30 }, () =>
      mkFrame({ shape: "indexOut", indexTip: { x: 0.5, y: 0.4 } }),
    );
    const r = simulate(new DynamicSignDetector(), frames);
    expect(r).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Negative cases & lifecycle
// ---------------------------------------------------------------------------

describe("DynamicSignDetector — false-positive guards", () => {
  it("a still pinky-out pose (letter I) doesn't fire J", () => {
    const frames = Array.from({ length: 40 }, () =>
      mkFrame({ shape: "pinkyOut", pinkyTip: { x: 0.55, y: 0.4 } }),
    );
    const r = simulate(new DynamicSignDetector(), frames);
    expect(r).toBeNull();
  });

  it("a still index-out pose (letter D) doesn't fire Z", () => {
    const frames = Array.from({ length: 40 }, () =>
      mkFrame({ shape: "indexOut", indexTip: { x: 0.55, y: 0.4 } }),
    );
    const r = simulate(new DynamicSignDetector(), frames);
    expect(r).toBeNull();
  });
});

describe("DynamicSignDetector — buffer/cooldown lifecycle", () => {
  it("resetting the buffer when no hand is visible", () => {
    const detector = new DynamicSignDetector();
    detector.push(mkFrame({ shape: "pinkyOut" }), 0);
    detector.push(mkFrame({ shape: "pinkyOut" }), 33);
    detector.push(null, 66);
    const stillFrames = Array.from({ length: 20 }, () =>
      mkFrame({ shape: "pinkyOut", pinkyTip: { x: 0.5, y: 0.4 } }),
    );
    const r = simulate(detector, stillFrames, 100);
    expect(r).toBeNull();
  });

  it("does not fire twice within the cooldown window", () => {
    const N = 25;
    const buildJ = () => {
      const frames: DetectedHand[] = [];
      for (let i = 0; i < N; i++) {
        const t = i / (N - 1);
        let x: number;
        let y: number;
        if (t < 0.6) {
          x = 0.55;
          y = lerp(0.3, 0.55, t / 0.6);
        } else {
          const tt = (t - 0.6) / 0.4;
          x = lerp(0.55, 0.45, tt);
          y = lerp(0.55, 0.62, tt);
        }
        frames.push(mkFrame({ shape: "pinkyOut", pinkyTip: { x, y } }));
      }
      return frames;
    };

    const detector = new DynamicSignDetector();
    const first = simulate(detector, buildJ(), 0);
    expect(first?.label).toBe("J");

    const second = simulate(detector, buildJ(), 25 * 33);
    expect(second).toBeNull();
  });
});
