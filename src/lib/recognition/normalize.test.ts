import { describe, expect, it } from "vitest";
import { normalizeHand, syntheticHand } from "./normalize";

describe("normalizeHand", () => {
  it("places the wrist at the origin", () => {
    const hand = syntheticHand({ origin: { x: 0.3, y: 0.7, z: 0.05 } });
    const { vector } = normalizeHand(hand, "Right");
    expect(vector[0]).toBeCloseTo(0, 6);
    expect(vector[1]).toBeCloseTo(0, 6);
    expect(vector[2]).toBeCloseTo(0, 6);
  });

  it("scales so wrist→middle-MCP distance equals 1", () => {
    const hand = syntheticHand({ scale: 0.02 });
    const { vector } = normalizeHand(hand, "Right");
    // Landmark 9 is at index 9 → vector slice [27, 28, 29]
    const dist = Math.hypot(vector[27], vector[28], vector[29]);
    expect(dist).toBeCloseTo(1, 6);
  });

  it("is translation invariant", () => {
    const a = normalizeHand(syntheticHand({ origin: { x: 0.1, y: 0.2, z: 0 } }), "Right");
    const b = normalizeHand(syntheticHand({ origin: { x: 0.8, y: 0.6, z: 0.1 } }), "Right");
    for (let i = 0; i < 63; i++) {
      expect(a.vector[i]).toBeCloseTo(b.vector[i], 6);
    }
  });

  it("is scale invariant", () => {
    const a = normalizeHand(syntheticHand({ scale: 0.005 }), "Right");
    const b = normalizeHand(syntheticHand({ scale: 0.05 }), "Right");
    for (let i = 0; i < 63; i++) {
      expect(a.vector[i]).toBeCloseTo(b.vector[i], 6);
    }
  });

  it("mirrors left hands by default so left/right map to same space", () => {
    const hand = syntheticHand();
    const right = normalizeHand(hand, "Right");
    const left = normalizeHand(hand, "Left");
    for (let i = 0; i < 21; i++) {
      // X is mirrored on left hands; Y and Z unchanged.
      expect(left.vector[i * 3]).toBeCloseTo(-right.vector[i * 3], 6);
      expect(left.vector[i * 3 + 1]).toBeCloseTo(right.vector[i * 3 + 1], 6);
      expect(left.vector[i * 3 + 2]).toBeCloseTo(right.vector[i * 3 + 2], 6);
    }
  });

  it("can opt out of mirroring", () => {
    const hand = syntheticHand();
    const right = normalizeHand(hand, "Right", { mirrorLeft: false });
    const left = normalizeHand(hand, "Left", { mirrorLeft: false });
    for (let i = 0; i < 63; i++) {
      expect(left.vector[i]).toBeCloseTo(right.vector[i], 6);
    }
  });

  it("rejects landmark arrays of the wrong length", () => {
    expect(() => normalizeHand([], "Right")).toThrow();
    expect(() => normalizeHand(syntheticHand().slice(0, 20), "Right")).toThrow();
  });
});
