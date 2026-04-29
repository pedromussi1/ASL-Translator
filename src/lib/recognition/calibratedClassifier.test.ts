import { beforeEach, describe, expect, it } from "vitest";
import {
  CALIBRATION_STORAGE_KEY,
  CalibratedClassifier,
  LetterRecorder,
  type CalibrationData,
} from "./calibratedClassifier";
import type { NormalizedHand } from "./types";

const hand = (vec: number[], handedness: "Left" | "Right" = "Right"): NormalizedHand => ({
  vector: Float32Array.from(vec.concat(Array(63 - vec.length).fill(0))),
  handedness,
});

beforeEach(() => {
  if (typeof localStorage !== "undefined") localStorage.clear();
});

describe("LetterRecorder", () => {
  it("averages frames into a mean prototype", () => {
    const r = new LetterRecorder("A");
    r.push(hand([1, 1, 1]));
    r.push(hand([3, 3, 3]));
    expect(r.count).toBe(2);
    const proto = r.buildPrototype();
    expect(proto).not.toBeNull();
    expect(proto!.slice(0, 3)).toEqual([2, 2, 2]);
  });

  it("returns null when nothing was recorded", () => {
    const r = new LetterRecorder("A");
    expect(r.buildPrototype()).toBeNull();
  });

  it("reset clears the buffer", () => {
    const r = new LetterRecorder("A");
    r.push(hand([1, 0, 0]));
    r.reset();
    expect(r.count).toBe(0);
    expect(r.buildPrototype()).toBeNull();
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

  it("confidence is higher when the runner-up is far away", () => {
    const c = CalibratedClassifier.fromData(data);
    const close = c.recognizeSync(hand([0.99, 0.01]));
    const ambiguous = c.recognizeSync(hand([0.5, 0.5]));
    expect(close.confidence).toBeGreaterThan(ambiguous.confidence);
  });

  it("returns confidence 0 when there are no prototypes", () => {
    const empty: CalibrationData = {
      version: 1,
      createdAt: 0,
      prototypes: {},
      sampleCounts: {},
    };
    const c = CalibratedClassifier.fromData(empty);
    expect(c.recognizeSync(hand([1, 0, 0])).confidence).toBe(0);
  });
});

describe("CalibratedClassifier.upsertLetter", () => {
  it("persists a new letter into a fresh calibration", () => {
    const proto = Array(63).fill(0.42);
    const data = CalibratedClassifier.upsertLetter("A", proto, 60);
    expect(data.prototypes.A).toEqual(proto);
    expect(data.sampleCounts.A).toBe(60);

    const loaded = CalibratedClassifier.loadDataFromStorage();
    expect(loaded?.prototypes.A).toEqual(proto);
  });

  it("replaces a letter without losing others", () => {
    CalibratedClassifier.upsertLetter("A", Array(63).fill(1), 60);
    CalibratedClassifier.upsertLetter("B", Array(63).fill(2), 60);
    const data = CalibratedClassifier.upsertLetter("A", Array(63).fill(3), 80);
    expect(data.prototypes.A[0]).toBe(3);
    expect(data.prototypes.B[0]).toBe(2);
    expect(data.sampleCounts.A).toBe(80);
    expect(data.sampleCounts.B).toBe(60);
  });

  it("removeLetter strips just that letter", () => {
    CalibratedClassifier.upsertLetter("A", Array(63).fill(1), 60);
    CalibratedClassifier.upsertLetter("B", Array(63).fill(2), 60);
    const data = CalibratedClassifier.removeLetter("A");
    expect(data?.prototypes.A).toBeUndefined();
    expect(data?.prototypes.B).toBeDefined();
  });

  it("loadFromStorage returns null when no letters are calibrated", () => {
    expect(CalibratedClassifier.loadFromStorage()).toBeNull();
    CalibratedClassifier.saveToStorage({
      version: 1,
      createdAt: 0,
      prototypes: {},
      sampleCounts: {},
    });
    expect(CalibratedClassifier.loadFromStorage()).toBeNull();
  });
});

describe("CalibratedClassifier storage round-trip", () => {
  it("saveToStorage + loadFromStorage preserves data", () => {
    const original: CalibrationData = {
      version: 1,
      createdAt: 12345,
      prototypes: { Q: Array(63).fill(0.5) },
      sampleCounts: { Q: 30 },
    };
    CalibratedClassifier.saveToStorage(original);
    const loaded = CalibratedClassifier.loadDataFromStorage();
    expect(loaded?.prototypes.Q.length).toBe(63);
    expect(loaded?.sampleCounts.Q).toBe(30);
    expect(localStorage.getItem(CALIBRATION_STORAGE_KEY)).toBeTruthy();
  });
});
