import type { Prediction } from "./types";

/**
 * Word buffer state machine. Translates a stream of per-frame predictions
 * (letter + confidence, or null for "no hand") into committed letters and
 * spoken words.
 *
 * Lifecycle of a single letter:
 *   IDLE
 *     ↓ confident prediction for letter L
 *   STABILIZING(L) — accumulating dwell time
 *     ↓ same L held for `stableHoldMs`
 *   COMMITTED — letter appended to currentWord
 *     ↓ stays in LETTER_LOCK(L) until either a different stable letter
 *       arrives (→ STABILIZING(L')) or the hand leaves the frame for
 *       `handAbsentTimeoutMs` (→ COMMIT_WORD, IDLE).
 *
 * The LETTER_LOCK avoids committing the same letter twice when the user
 * keeps their hand still — the hand must change letters or drop out of
 * frame to "release" it.
 */

export interface WordBufferConfig {
  /** ms a letter must be confidently held before it commits. */
  stableHoldMs: number;
  /** ms with no hand visible after a letter commits before we end the word. */
  handAbsentTimeoutMs: number;
  /**
   * ms of absence required to "release" a locked letter so the same letter
   * can be re-committed. Shorter than handAbsentTimeoutMs so the user can
   * sign repeated letters (HELLO, BOOK) by briefly dropping the hand
   * without losing the word.
   */
  letterReleaseMs: number;
  /** Predictions below this confidence are ignored (treated as no-hand). */
  confidenceThreshold: number;
  /** Lowercase letters in committed text. */
  lowercase: boolean;
}

export const DEFAULT_WORD_BUFFER_CONFIG: WordBufferConfig = {
  stableHoldMs: 300,
  handAbsentTimeoutMs: 800,
  letterReleaseMs: 200,
  confidenceThreshold: 0.7,
  lowercase: false,
};

export type WordBufferEvent =
  | { type: "letter_committed"; letter: string; word: string; at: number }
  | { type: "word_committed"; word: string; at: number }
  | { type: "tentative_change"; letter: string | null; confidence: number; at: number }
  | { type: "cleared"; at: number };

type State =
  | { kind: "idle"; lastHandSeenAt: number | null }
  | { kind: "stabilizing"; letter: string; since: number; lastSeenAt: number }
  | {
      kind: "letter_lock";
      letter: string;
      lastSeenAt: number;
      /** First frame at which we saw "no hand" since the lock began, or null. */
      absentSince: number | null;
    };

export class WordBuffer {
  private state: State = { kind: "idle", lastHandSeenAt: null };
  private word = "";
  private listeners = new Set<(e: WordBufferEvent) => void>();
  private lastTentative: string | null = null;
  private readonly config: WordBufferConfig;

  constructor(config: Partial<WordBufferConfig> = {}) {
    this.config = { ...DEFAULT_WORD_BUFFER_CONFIG, ...config };
  }

  /**
   * Feed one frame's prediction. Pass `null` when no hand was detected
   * (or a hand was detected but the prediction was below threshold).
   */
  feed(prediction: Prediction | null, at: number): void {
    const accepted =
      prediction && prediction.confidence >= this.config.confidenceThreshold
        ? prediction
        : null;

    this.publishTentative(accepted, at);

    switch (this.state.kind) {
      case "idle":
        this.handleIdle(accepted, at);
        return;
      case "stabilizing":
        this.handleStabilizing(accepted, at);
        return;
      case "letter_lock":
        this.handleLetterLock(accepted, at);
        return;
    }
  }

  private handleIdle(p: Prediction | null, at: number): void {
    if (this.state.kind !== "idle") return;
    if (p) {
      this.state = { kind: "stabilizing", letter: p.label, since: at, lastSeenAt: at };
      return;
    }
    // No hand: nothing to do unless we're still building a word and
    // the absent timeout has elapsed since the last hand was seen.
    if (this.word.length > 0 && this.state.lastHandSeenAt !== null) {
      if (at - this.state.lastHandSeenAt >= this.config.handAbsentTimeoutMs) {
        this.commitWord(at);
      }
    }
  }

  private handleStabilizing(p: Prediction | null, at: number): void {
    if (this.state.kind !== "stabilizing") return;
    const s = this.state;
    if (p && p.label === s.letter) {
      // Held same letter: check dwell.
      if (at - s.since >= this.config.stableHoldMs) {
        this.commitLetter(s.letter, at);
        this.state = {
          kind: "letter_lock",
          letter: s.letter,
          lastSeenAt: at,
          absentSince: null,
        };
      } else {
        this.state = { ...s, lastSeenAt: at };
      }
      return;
    }
    if (p && p.label !== s.letter) {
      // Switched letter: restart stabilization.
      this.state = { kind: "stabilizing", letter: p.label, since: at, lastSeenAt: at };
      return;
    }
    // No (accepted) prediction: bail back to idle, but remember last hand-seen
    // time so the word commits if absence persists.
    this.state = { kind: "idle", lastHandSeenAt: s.lastSeenAt };
    this.handleIdle(null, at);
  }

  private handleLetterLock(p: Prediction | null, at: number): void {
    if (this.state.kind !== "letter_lock") return;
    const s = this.state;
    if (p && p.label !== s.letter) {
      // Different letter incoming — start stabilizing it.
      this.state = { kind: "stabilizing", letter: p.label, since: at, lastSeenAt: at };
      return;
    }
    if (p && p.label === s.letter) {
      // Same letter. If we'd seen a meaningful absence, treat this as a
      // re-press: re-stabilize so the letter can commit again.
      if (
        s.absentSince !== null &&
        at - s.absentSince >= this.config.letterReleaseMs
      ) {
        this.state = { kind: "stabilizing", letter: p.label, since: at, lastSeenAt: at };
        return;
      }
      // Otherwise just refresh and clear absentSince.
      this.state = { ...s, lastSeenAt: at, absentSince: null };
      return;
    }
    // No accepted prediction.
    const absentSince = s.absentSince ?? at;
    if (at - s.lastSeenAt >= this.config.handAbsentTimeoutMs) {
      this.commitWord(at);
      this.state = { kind: "idle", lastHandSeenAt: null };
      return;
    }
    this.state = { ...s, absentSince };
  }

  private commitLetter(letter: string, at: number): void {
    const ch = this.config.lowercase ? letter.toLowerCase() : letter.toUpperCase();
    this.word += ch;
    this.emit({ type: "letter_committed", letter: ch, word: this.word, at });
  }

  private commitWord(at: number): void {
    if (!this.word) return;
    const word = this.word;
    this.word = "";
    this.emit({ type: "word_committed", word, at });
  }

  private publishTentative(p: Prediction | null, at: number): void {
    const next = p?.label ?? null;
    if (next !== this.lastTentative) {
      this.lastTentative = next;
      this.emit({
        type: "tentative_change",
        letter: next,
        confidence: p?.confidence ?? 0,
        at,
      });
    }
  }

  private emit(e: WordBufferEvent): void {
    for (const l of this.listeners) l(e);
  }

  on(listener: (e: WordBufferEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Discard the in-progress word without speaking it. */
  clear(at: number = performance.now()): void {
    if (this.word) this.word = "";
    this.state = { kind: "idle", lastHandSeenAt: null };
    this.lastTentative = null;
    this.emit({ type: "cleared", at });
  }

  /** Drop the last committed letter from the in-progress word. */
  backspace(): void {
    if (this.word.length === 0) return;
    this.word = this.word.slice(0, -1);
  }

  get currentWord(): string {
    return this.word;
  }

  get tentativeLetter(): string | null {
    return this.lastTentative;
  }

  /** Snapshot of state, intended for tests / debugging only. */
  get debugState(): Readonly<State> {
    return this.state;
  }
}
