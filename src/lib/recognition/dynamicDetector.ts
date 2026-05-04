import type { DetectedHand, Point3 } from "./types";

export type DynamicLabel = "J" | "Z";

export interface DynamicResult {
  label: DynamicLabel;
  confidence: number;
}

export interface MatcherScores {
  J: number;
  Z: number;
}

interface FrameFeatures {
  timestampMs: number;
  fingers: { thumb: number; index: number; middle: number; ring: number; pinky: number };
  wrist: Point3;
  indexTip: Point3;
  pinkyTip: Point3;
}

const BUFFER_MS = 1500;
const MIN_FRAMES_FOR_DETECTION = 12;
const COOLDOWN_MS = 1500;
const FIRE_THRESHOLD = 0.7;

/** Landmark indices on a 21-point MediaPipe hand. */
const FINGER_LM = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
} as const;

// Real human hands rarely produce a "perfectly straight" finger; an extension
// threshold of 0.92 was too strict and matchers were silently failing.
const EXT_THRESH = 0.85;
const CURL_THRESH = 0.75;

/**
 * Heuristic detector for the two dynamic signs (J and Z).
 *
 * Each frame is buffered with pre-computed shape features; on every push we
 * evaluate both matchers and surface the highest-confidence hit above
 * `FIRE_THRESHOLD`. A short cooldown prevents repeat firings while the user
 * naturally returns to a neutral pose.
 *
 * Both signs require an unambiguous handshape held throughout the buffer
 * AND a specific motion signature, so static fingerspelling poses (an "I"
 * held still, a "D" held still) don't false-fire.
 */
export class DynamicSignDetector {
  private buffer: FrameFeatures[] = [];
  private cooldownUntil = 0;
  private lastScores: MatcherScores = { J: 0, Z: 0 };

  push(hand: DetectedHand | null, timestampMs: number): DynamicResult | null {
    // Drop frames with no hand — they break trajectory continuity.
    if (!hand) {
      this.buffer = [];
      return null;
    }

    const features = computeFeatures(hand, timestampMs);
    this.buffer.push(features);

    // Evict frames older than BUFFER_MS.
    const cutoff = timestampMs - BUFFER_MS;
    while (this.buffer.length > 0 && this.buffer[0].timestampMs < cutoff) {
      this.buffer.shift();
    }

    if (this.buffer.length < MIN_FRAMES_FOR_DETECTION) return null;

    const j = matchJ(this.buffer, hand.handedness);
    const z = matchZ(this.buffer, hand.handedness);
    this.lastScores = { J: j, Z: z };

    if (timestampMs < this.cooldownUntil) return null;

    const candidates: DynamicResult[] = [
      { label: "J", confidence: j },
      { label: "Z", confidence: z },
    ];
    candidates.sort((a, b) => b.confidence - a.confidence);
    const best = candidates[0];
    if (best.confidence < FIRE_THRESHOLD) return null;

    this.cooldownUntil = timestampMs + COOLDOWN_MS;
    this.buffer = []; // start fresh after a fire
    return best;
  }

  /** Latest per-matcher scores; useful for a debug HUD. */
  getLastScores(): MatcherScores {
    return { ...this.lastScores };
  }

  reset(): void {
    this.buffer = [];
    this.cooldownUntil = 0;
    this.lastScores = { J: 0, Z: 0 };
  }
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

function computeFeatures(hand: DetectedHand, timestampMs: number): FrameFeatures {
  const lms = hand.landmarks;
  return {
    timestampMs,
    fingers: {
      thumb: fingerExtension(lms, FINGER_LM.thumb),
      index: fingerExtension(lms, FINGER_LM.index),
      middle: fingerExtension(lms, FINGER_LM.middle),
      ring: fingerExtension(lms, FINGER_LM.ring),
      pinky: fingerExtension(lms, FINGER_LM.pinky),
    },
    wrist: lms[0],
    indexTip: lms[8],
    pinkyTip: lms[20],
  };
}

/**
 * 1.0 = perfectly straight finger, ~0.6-0.7 = curled. Computed as the ratio of
 * straight-line distance from base to tip vs the kinked piecewise distance
 * through PIP/DIP. Robust to hand size since both legs are in the same scale.
 */
function fingerExtension(lms: Point3[], idxs: readonly number[]): number {
  const [a, b, c, d] = idxs.map((i) => lms[i]);
  const straight = dist3(a, d);
  const kinked = dist3(a, b) + dist3(b, c) + dist3(c, d);
  if (kinked < 1e-6) return 0;
  return straight / kinked;
}

function dist3(p: Point3, q: Point3): number {
  const dx = p.x - q.x;
  const dy = p.y - q.y;
  const dz = p.z - q.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

function shapePinkyOnly(f: FrameFeatures): number {
  const pinky = f.fingers.pinky;
  const others = [f.fingers.index, f.fingers.middle, f.fingers.ring];
  if (pinky < EXT_THRESH) return 0;
  if (others.some((x) => x > CURL_THRESH)) return 0;
  return clamp01((pinky - EXT_THRESH) / (1 - EXT_THRESH));
}

function shapeIndexOnly(f: FrameFeatures): number {
  const index = f.fingers.index;
  const others = [f.fingers.middle, f.fingers.ring, f.fingers.pinky];
  if (index < EXT_THRESH) return 0;
  if (others.some((x) => x > CURL_THRESH)) return 0;
  return clamp01((index - EXT_THRESH) / (1 - EXT_THRESH));
}

/** Average shape score across all frames (must hold the shape throughout). */
function avgShape(buf: FrameFeatures[], shape: (f: FrameFeatures) => number): number {
  let sum = 0;
  for (const f of buf) sum += shape(f);
  return sum / buf.length;
}

// J — pinky extended, traces a J: starts upper-right, ends lower-left.
function matchJ(buf: FrameFeatures[], handedness: "Left" | "Right"): number {
  const shape = avgShape(buf, shapePinkyOnly);
  if (shape < 0.4) return 0;

  const xs = buf.map((f) => f.pinkyTip.x);
  const ys = buf.map((f) => f.pinkyTip.y);
  const xRange = Math.max(...xs) - Math.min(...xs);
  const yRange = Math.max(...ys) - Math.min(...ys);
  if (yRange < 0.08 || xRange < 0.04) return 0;

  // The video is mirrored on screen — for a right-handed user signing J, on
  // screen the hand moves left→right but in MediaPipe's image coords (which
  // are NOT mirrored) the pinky goes RIGHT then DOWN-LEFT.
  const earlyX = avg(xs.slice(0, Math.floor(xs.length / 4)));
  const lateX = avg(xs.slice(-Math.floor(xs.length / 4)));
  const earlyY = avg(ys.slice(0, Math.floor(ys.length / 4)));
  const lateY = avg(ys.slice(-Math.floor(ys.length / 4)));

  const movedDown = lateY > earlyY ? 1 : 0;
  const expectedXMoves = handedness === "Right" ? lateX < earlyX : lateX > earlyX;
  const motionScore =
    movedDown * (Math.min(yRange, 0.3) / 0.3) * (expectedXMoves ? 1 : 0.4);

  return clamp01(shape * 0.4 + motionScore * 0.8);
}

// Z — index extended, traces three segments: right, diagonal-down-left, right.
function matchZ(buf: FrameFeatures[], handedness: "Left" | "Right"): number {
  const shape = avgShape(buf, shapeIndexOnly);
  if (shape < 0.4) return 0;

  const n = buf.length;
  const k = [
    buf[Math.floor(n * 0.05)],
    buf[Math.floor(n * 0.35)],
    buf[Math.floor(n * 0.65)],
    buf[n - 1],
  ];
  const x = k.map((f) => f.indexTip.x);
  const y = k.map((f) => f.indexTip.y);

  const mirror = handedness === "Right" ? -1 : 1;
  const dx1 = (x[1] - x[0]) * mirror; // expect rightward
  const dx2 = (x[2] - x[1]) * mirror; // expect leftward
  const dx3 = (x[3] - x[2]) * mirror; // expect rightward
  const dy2 = y[2] - y[1]; // expect downward (positive y is down)

  const xRange = Math.max(...x) - Math.min(...x);
  const yRange = Math.max(...y) - Math.min(...y);
  if (xRange < 0.06 || yRange < 0.04) return 0;

  const s1 = clamp01(dx1 / 0.05);
  const s2 = clamp01(-dx2 / 0.05) * clamp01(dy2 / 0.04);
  const s3 = clamp01(dx3 / 0.05);
  const traj = (s1 + s2 + s3) / 3;
  return clamp01(shape * 0.3 + traj * 0.85);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function avg(xs: number[]): number {
  let s = 0;
  for (const x of xs) s += x;
  return xs.length ? s / xs.length : 0;
}
