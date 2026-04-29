import { describe, expect, it } from "vitest";
import {
  CalibratedClassifier,
  CalibrationSession,
  type CalibrationData,
} from "./calibratedClassifier";
import type { NormalizedHand } from "./types";

const hand = (vec: number[], handedness: "Left" | "Right" = "Right"): NormalizedHand => ({
  vector: Float32Array.from(vec.concat(Array(63 - vec.length).fill(0))),
  handedness,
});

describe("CalibrationSession", () => {
  it("collects frames per letter and averages them into a prototype", () => {
    const s = new CalibrationSession(["A", "B"]);
    s.push("A", hand([1, 1, 1]));
    s.push("A", hand([3, 3, 3]));
    expect(s.recordedCount("A")).toBe(2);

    const data = s.build();
    expect(data.version).toBe(1);
    expect(data.prototypes.A.slice(0, 3)).toEqual([2, 2, 2]);
    expect(data.sampleCounts.A).toBe(2);
    expect(data.prototypes.B).toBeUndefined();
  });

  it("reset() drops samples for a letter", () => {
    const s = new CalibrationSession(["A"]);
    s.push("A", hand([1, 0, 0]));
    s.reset("A");
    expect(s.recordedCount("A")).toBe(0);
    expect(s.build().prototypes.A).toBeUndefined();
  });
});

describe("CalibratedClassifier", () => {
  const data: CalibrationData = {
    version: 1,
    createdAt: 0,
    prototypes: {
      A: Array(63).fill(0).map((_, i) => (i === 0 ? 1 : 0)),
      B: Array(63).fill(0).map((_, i) => (i === 0 ? -1 : 0)),
      C: Array(63).fill(0).map((_, i) => (i === 1 ? 1 : 0)),
    },
    sampleCounts: { A: 60, B: 60, C: 60 },
  };

  it("returns the nearest prototype label", () => {
    const c = CalibratedClassifier.fromData(data);
    const r = c.recognizeSync(hand([0.9, 0.05]));
    expect(r.label).toBe("A");
    expect(r.confidence).toBeGreaterThan(0);
  });

  it("confidence is high when the runner-up is far away", () => {
    const c = CalibratedClassifier.fromData(data);
    const close = c.recognizeSync(hand([0.99, 0.01]));
    const ambiguous = c.recognizeSync(hand([0.5, 0.5]));
    expect(close.confidence).toBeGreaterThan(ambiguous.confidence);
  });

  it("returns confidence 0 for an empty calibration", () => {
    const empty: CalibrationData = {
      version: 1,
      createdAt: 0,
      prototypes: {},
      sampleCounts: {},
    };
    const c = CalibratedClassifier.fromData(empty);
    const r = c.recognizeSync(hand([1, 0, 0]));
    expect(r.confidence).toBe(0);
  });

  it("confidence ranges in [0, 1]", () => {
    const c = CalibratedClassifier.fromData(data);
    for (let i = 0; i < 5; i++) {
      const r = c.recognizeSync(hand([Math.random(), Math.random(), Math.random()]));
      expect(r.confidence).toBeGreaterThanOrEqual(0);
      expect(r.confidence).toBeLessThanOrEqual(1);
    }
  });
});
