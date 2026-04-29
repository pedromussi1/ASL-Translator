import { describe, expect, it } from "vitest";
import { PredictionSmoother } from "./smoother";

const p = (label: string, confidence: number) => ({ label, confidence });

describe("PredictionSmoother", () => {
  it("returns the dominant label across a window", () => {
    const s = new PredictionSmoother(5);
    s.push(p("A", 0.9));
    s.push(p("A", 0.85));
    s.push(p("T", 0.6));
    s.push(p("A", 0.95));
    const out = s.push(p("A", 0.88));
    expect(out?.label).toBe("A");
    expect(out?.confidence).toBeGreaterThan(0.85);
  });

  it("filters single-frame flicker", () => {
    const s = new PredictionSmoother(5);
    s.push(p("S", 0.8));
    s.push(p("S", 0.85));
    s.push(p("M", 0.55)); // single noisy frame
    s.push(p("S", 0.88));
    const out = s.push(p("S", 0.9));
    expect(out?.label).toBe("S");
  });

  it("treats nulls as a class — silence wins when most frames are null", () => {
    const s = new PredictionSmoother(5);
    s.push(null);
    s.push(null);
    s.push(p("X", 0.95)); // hand momentarily detected
    s.push(null);
    const out = s.push(null);
    expect(out).toBeNull();
  });

  it("emits the live prediction immediately while warming up", () => {
    const s = new PredictionSmoother(5);
    const first = s.push(p("B", 0.91));
    expect(first?.label).toBe("B");
  });

  it("ties break toward the more confident label", () => {
    const s = new PredictionSmoother(4);
    s.push(p("A", 0.7));
    s.push(p("A", 0.7));
    s.push(p("B", 0.95));
    const out = s.push(p("B", 0.95));
    expect(out?.label).toBe("B");
  });

  it("reset clears the window", () => {
    const s = new PredictionSmoother(3);
    s.push(p("A", 0.9));
    s.push(p("A", 0.9));
    s.reset();
    const out = s.push(p("Z", 0.8));
    expect(out?.label).toBe("Z");
  });
});
