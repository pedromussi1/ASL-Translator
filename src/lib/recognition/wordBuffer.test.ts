import { describe, expect, it, vi } from "vitest";
import { WordBuffer, type WordBufferEvent } from "./wordBuffer";

const defaults = { stableHoldMs: 300, handAbsentTimeoutMs: 800 };
const conf = (label: string, confidence = 0.95) => ({ label, confidence });

function captureEvents(buf: WordBuffer): WordBufferEvent[] {
  const events: WordBufferEvent[] = [];
  buf.on((e) => events.push(e));
  return events;
}

describe("WordBuffer", () => {
  it("commits a letter once the same prediction is held for stableHoldMs", () => {
    const buf = new WordBuffer(defaults);
    const events = captureEvents(buf);

    buf.feed(conf("A"), 0);
    buf.feed(conf("A"), 100);
    buf.feed(conf("A"), 250);
    expect(buf.currentWord).toBe("");

    buf.feed(conf("A"), 350); // crosses 300ms threshold
    expect(buf.currentWord).toBe("A");
    expect(events.some((e) => e.type === "letter_committed" && e.letter === "A")).toBe(true);
  });

  it("does not double-commit the same letter while held", () => {
    const buf = new WordBuffer(defaults);
    buf.feed(conf("A"), 0);
    buf.feed(conf("A"), 350);
    expect(buf.currentWord).toBe("A");
    buf.feed(conf("A"), 1000); // still holding
    buf.feed(conf("A"), 2000);
    expect(buf.currentWord).toBe("A");
  });

  it("commits a different letter when the user changes hand shape and holds", () => {
    const buf = new WordBuffer(defaults);
    buf.feed(conf("H"), 0);
    buf.feed(conf("H"), 350); // commit H
    buf.feed(conf("E"), 400);
    buf.feed(conf("E"), 750); // commit E
    expect(buf.currentWord).toBe("HE");
  });

  it("requires re-stabilization for repeated letters separated by hand-absence", () => {
    const buf = new WordBuffer(defaults);
    // Spell HELLO: H-E-L-L-O. The trick: same letter twice in a row.
    buf.feed(conf("H"), 0);
    buf.feed(conf("H"), 350); // H
    buf.feed(conf("E"), 400);
    buf.feed(conf("E"), 750); // HE
    buf.feed(conf("L"), 800);
    buf.feed(conf("L"), 1150); // HEL
    // Hand drops briefly between the two L's — within absent timeout so word doesn't end.
    buf.feed(null, 1200);
    buf.feed(null, 1400);
    buf.feed(conf("L"), 1500);
    buf.feed(conf("L"), 1850); // HELL
    buf.feed(conf("O"), 1900);
    buf.feed(conf("O"), 2250); // HELLO
    expect(buf.currentWord).toBe("HELLO");
  });

  it("ends the word when the hand is absent past handAbsentTimeoutMs", () => {
    const buf = new WordBuffer(defaults);
    const events = captureEvents(buf);
    buf.feed(conf("H"), 0);
    buf.feed(conf("H"), 350);
    buf.feed(conf("I"), 400);
    buf.feed(conf("I"), 750);
    expect(buf.currentWord).toBe("HI");

    // Hand drops; we feed nulls past the absent timeout (800ms after last seen).
    buf.feed(null, 800);
    buf.feed(null, 1200);
    buf.feed(null, 1700); // 750 + 800 = 1550, this is past it
    expect(buf.currentWord).toBe("");
    const wordCommit = events.find((e) => e.type === "word_committed");
    expect(wordCommit).toMatchObject({ type: "word_committed", word: "HI" });
  });

  it("ignores predictions below the confidence threshold", () => {
    const buf = new WordBuffer({ ...defaults, confidenceThreshold: 0.7 });
    buf.feed(conf("A", 0.5), 0);
    buf.feed(conf("A", 0.5), 350);
    expect(buf.currentWord).toBe("");
  });

  it("emits tentative_change events as the live letter changes", () => {
    const buf = new WordBuffer(defaults);
    const events = captureEvents(buf);
    buf.feed(conf("A"), 0);
    buf.feed(conf("A"), 100);
    buf.feed(conf("B"), 150);
    buf.feed(null, 200);

    const tentatives = events
      .filter((e): e is Extract<WordBufferEvent, { type: "tentative_change" }> => e.type === "tentative_change")
      .map((e) => e.letter);
    expect(tentatives).toEqual(["A", "B", null]);
  });

  it("clear() drops the in-progress word and resets state without speaking", () => {
    const buf = new WordBuffer(defaults);
    const speakSpy = vi.fn();
    buf.on((e) => {
      if (e.type === "word_committed") speakSpy(e.word);
    });
    buf.feed(conf("X"), 0);
    buf.feed(conf("X"), 350);
    expect(buf.currentWord).toBe("X");
    buf.clear(400);
    expect(buf.currentWord).toBe("");
    expect(speakSpy).not.toHaveBeenCalled();
  });

  it("backspace() removes the last letter only", () => {
    const buf = new WordBuffer(defaults);
    buf.feed(conf("A"), 0);
    buf.feed(conf("A"), 350);
    buf.feed(conf("B"), 400);
    buf.feed(conf("B"), 750);
    expect(buf.currentWord).toBe("AB");
    buf.backspace();
    expect(buf.currentWord).toBe("A");
    buf.backspace();
    expect(buf.currentWord).toBe("");
    buf.backspace(); // no-op on empty
    expect(buf.currentWord).toBe("");
  });

  it("does not commit a word that was never started (empty buffer + absent)", () => {
    const buf = new WordBuffer(defaults);
    const events = captureEvents(buf);
    buf.feed(null, 0);
    buf.feed(null, 1000);
    expect(events.some((e) => e.type === "word_committed")).toBe(false);
  });
});
