import type { DetectedFace, DetectedHand, Point3 } from "./types";

export type DynamicLabel = "J" | "Z" | "YES" | "HELLO" | "THANK YOU";

export interface DynamicResult {
  label: DynamicLabel;
  confidence: number;
}

interface FrameFeatures {
  timestampMs: number;
  fingers: { thumb: number; index: number; middle: number; ring: number; pinky: number };
  wrist: Point3;
  indexTip: Point3;
  pinkyTip: Point3;
  face: DetectedFace | null;
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

/**
 * Heuristic detector for the five dynamic signs (J, Z, YES, HELLO, THANK YOU).
 * Each frame is buffered with pre-computed shape features; on every push we
 * evaluate all matchers and surface the highest-confidence hit above
 * `FIRE_THRESHOLD`. A short cooldown prevents repeat firings while the user
 * naturally returns to a neutral pose.
 *
 * J and Z and YES have unambiguous motion signatures — these are reliable.
 * HELLO and THANK YOU motions overlap (both are an open-hand outward sweep)
 * and ideally need face/pose landmarks to distinguish; we score them
 * conservatively until that pipeline is added.
 */
export class DynamicSignDetector {
  private buffer: FrameFeatures[] = [];
  private cooldownUntil = 0;

  push(
    hand: DetectedHand | null,
    face: DetectedFace | null,
    timestampMs: number,
  ): DynamicResult | null {
    // Drop frames with no hand — they break trajectory continuity.
    if (!hand) {
      this.buffer = [];
      return null;
    }

    const features = computeFeatures(hand, face, timestampMs);
    this.buffer.push(features);

    // Evict frames older than BUFFER_MS.
    const cutoff = timestampMs - BUFFER_MS;
    while (this.buffer.length > 0 && this.buffer[0].timestampMs < cutoff) {
      this.buffer.shift();
    }

    if (timestampMs < this.cooldownUntil) return null;
    if (this.buffer.length < MIN_FRAMES_FOR_DETECTION) return null;

    const candidates: DynamicResult[] = [
      { label: "J", confidence: matchJ(this.buffer, hand.handedness) },
      { label: "Z", confidence: matchZ(this.buffer, hand.handedness) },
      { label: "YES", confidence: matchYes(this.buffer) },
      { label: "HELLO", confidence: matchHello(this.buffer, hand.handedness) },
      { label: "THANK YOU", confidence: matchThankYou(this.buffer) },
    ];
    candidates.sort((a, b) => b.confidence - a.confidence);
    const best = candidates[0];
    if (best.confidence < FIRE_THRESHOLD) return null;

    this.cooldownUntil = timestampMs + COOLDOWN_MS;
    this.buffer = []; // start fresh after a fire
    return best;
  }

  reset(): void {
    this.buffer = [];
    this.cooldownUntil = 0;
  }
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

function computeFeatures(
  hand: DetectedHand,
  face: DetectedFace | null,
  timestampMs: number,
): FrameFeatures {
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
    face,
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

const EXT_THRESH = 0.92;
const CURL_THRESH = 0.83;

function shapePinkyOnly(f: FrameFeatures): number {
  // Pinky extended, index/middle/ring curled. Thumb is ambiguous in J handshape.
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

function shapeFist(f: FrameFeatures): number {
  const all = [f.fingers.index, f.fingers.middle, f.fingers.ring, f.fingers.pinky];
  if (all.some((x) => x > CURL_THRESH)) return 0;
  // The lower the max extension, the tighter the fist.
  const maxExt = Math.max(...all);
  return clamp01(1 - (maxExt - 0.6) / (CURL_THRESH - 0.6));
}

function shapeOpenHand(f: FrameFeatures): number {
  const all = [f.fingers.index, f.fingers.middle, f.fingers.ring, f.fingers.pinky];
  if (all.some((x) => x < EXT_THRESH)) return 0;
  return Math.min(...all.map((x) => clamp01((x - EXT_THRESH) / (1 - EXT_THRESH))));
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

  // The video is mirrored — for a right-handed user signing J, on-screen the
  // hand actually moves left→right but in MediaPipe's image coords (which are
  // not mirrored) the pinky goes RIGHT then DOWN-LEFT. Same direction logic
  // works for both hands once we account for which is "outside".
  const earlyX = avg(xs.slice(0, Math.floor(xs.length / 4)));
  const lateX = avg(xs.slice(-Math.floor(xs.length / 4)));
  const earlyY = avg(ys.slice(0, Math.floor(ys.length / 4)));
  const lateY = avg(ys.slice(-Math.floor(ys.length / 4)));

  const movedDown = lateY > earlyY ? 1 : 0;
  // J ends curving toward thumb side. For right hand on screen (= "Left" in
  // MediaPipe's labeling because image isn't flipped) the curve goes left.
  const expectedXMoves = handedness === "Right" ? lateX < earlyX : lateX > earlyX;
  const motionScore =
    movedDown * (Math.min(yRange, 0.3) / 0.3) * (expectedXMoves ? 1 : 0.4);

  return clamp01(shape * 0.4 + motionScore * 0.8);
}

// Z — index extended, traces three segments: right, diagonal-down-left, right.
function matchZ(buf: FrameFeatures[], handedness: "Left" | "Right"): number {
  const shape = avgShape(buf, shapeIndexOnly);
  if (shape < 0.4) return 0;

  // Sample 4 evenly-spaced keyframes.
  const n = buf.length;
  const k = [
    buf[Math.floor(n * 0.05)],
    buf[Math.floor(n * 0.35)],
    buf[Math.floor(n * 0.65)],
    buf[n - 1],
  ];
  const x = k.map((f) => f.indexTip.x);
  const y = k.map((f) => f.indexTip.y);

  // Sign-flip x deltas if user's right hand is on the screen-mirrored side.
  const mirror = handedness === "Right" ? -1 : 1;
  const dx1 = (x[1] - x[0]) * mirror; // expect rightward
  const dx2 = (x[2] - x[1]) * mirror; // expect leftward
  const dx3 = (x[3] - x[2]) * mirror; // expect rightward
  const dy2 = y[2] - y[1]; // expect downward (positive y is down)

  const xRange = Math.max(...x) - Math.min(...x);
  const yRange = Math.max(...y) - Math.min(...y);
  if (xRange < 0.06 || yRange < 0.04) return 0;

  // Each segment contributes proportionally to its directional correctness.
  const s1 = clamp01(dx1 / 0.05);
  const s2 = clamp01(-dx2 / 0.05) * clamp01(dy2 / 0.04);
  const s3 = clamp01(dx3 / 0.05);
  const traj = (s1 + s2 + s3) / 3;
  return clamp01(shape * 0.3 + traj * 0.85);
}

// YES — fist with vertical oscillation (bobbing up and down).
function matchYes(buf: FrameFeatures[]): number {
  const shape = avgShape(buf, shapeFist);
  if (shape < 0.4) return 0;

  const ys = buf.map((f) => f.wrist.y);
  const range = Math.max(...ys) - Math.min(...ys);
  if (range < 0.03) return 0;

  // Count direction reversals of dy as a proxy for oscillation.
  let reversals = 0;
  let prevSign = 0;
  for (let i = 1; i < ys.length; i++) {
    const d = ys[i] - ys[i - 1];
    if (Math.abs(d) < 0.002) continue;
    const s = d > 0 ? 1 : -1;
    if (prevSign !== 0 && s !== prevSign) reversals++;
    prevSign = s;
  }
  if (reversals < 2) return 0;

  const oscScore = clamp01(reversals / 4);
  const ampScore = clamp01(range / 0.08);
  return clamp01(shape * 0.4 + oscScore * 0.5 + ampScore * 0.3);
}

// HELLO — open hand at the temple, salute-like sweep outward. With face
// landmarks we anchor "near temple" to the actual ear/eye position rather
// than guessing from absolute screen y.
function matchHello(buf: FrameFeatures[], handedness: "Left" | "Right"): number {
  const shape = avgShape(buf, shapeOpenHand);
  if (shape < 0.5) return 0;

  const start = buf[0].wrist;
  const end = buf[buf.length - 1].wrist;
  const face = buf[0].face;

  const mirror = handedness === "Right" ? -1 : 1;
  const dx = (end.x - start.x) * mirror;
  if (dx < 0.05) return 0;

  let proximityScore: number;
  if (face) {
    const faceHeight = faceVerticalSize(face);
    const eyeY =
      face.rightEye?.y ?? face.leftEye?.y ?? face.noseTip?.y ?? null;
    const sideEar = handedness === "Right" ? face.rightEar : face.leftEar;
    const anchorEyeY = eyeY ?? 0.4;
    const dyToEye = Math.abs(start.y - anchorEyeY);
    const verticalCloseness = clamp01(1 - dyToEye / Math.max(faceHeight, 0.1));
    if (verticalCloseness < 0.3) return 0;

    let lateralScore = 0.5;
    if (sideEar) {
      // For a right-handed user, the right ear is on the LEFT of the
      // unmirrored frame. The hand starts at or just outside the same-side ear.
      const earDelta = (sideEar.x - start.x) * mirror;
      lateralScore = clamp01(0.5 + earDelta * 4);
    }
    proximityScore = verticalCloseness * 0.6 + lateralScore * 0.4;
  } else {
    if (start.y > 0.55) return 0;
    proximityScore = 0.4;
  }

  const motion = clamp01(dx / 0.18);
  const score = motion * proximityScore * clamp01(shape);
  return clamp01(face ? score : Math.min(0.7, score));
}

// THANK YOU — open hand starts at the chin/mouth, palm toward signer, sweeps
// outward and slightly down. With face landmarks we anchor "near mouth" to
// the actual mouth keypoint.
function matchThankYou(buf: FrameFeatures[]): number {
  const shape = avgShape(buf, shapeOpenHand);
  if (shape < 0.5) return 0;

  const start = buf[0].wrist;
  const end = buf[buf.length - 1].wrist;
  const face = buf[0].face;

  const yMove = end.y - start.y;
  if (yMove < 0.04) return 0;
  const xMove = Math.abs(end.x - start.x);

  let proximityScore: number;
  if (face) {
    const mouth = face.mouthCenter ?? face.noseTip ?? null;
    if (!mouth) {
      proximityScore = 0.4;
    } else {
      const faceHeight = faceVerticalSize(face);
      const dy = Math.abs(start.y - mouth.y);
      const dx = Math.abs(start.x - mouth.x);
      proximityScore =
        clamp01(1 - dy / Math.max(faceHeight, 0.1)) *
        clamp01(1 - dx / Math.max(faceHeight, 0.1));
      if (proximityScore < 0.3) return 0;
    }
  } else {
    if (start.y > 0.5) return 0;
    proximityScore = 0.4;
  }

  const motion = clamp01(yMove / 0.15) * clamp01((yMove + xMove) / 0.2);
  const score = motion * proximityScore * clamp01(shape);
  return clamp01(face ? score : Math.min(0.7, score));
}

/** Approximate face vertical span in normalized coords using available keypoints. */
function faceVerticalSize(face: DetectedFace): number {
  const eyeY = face.rightEye?.y ?? face.leftEye?.y ?? null;
  const mouthY = face.mouthCenter?.y ?? null;
  if (eyeY !== null && mouthY !== null) {
    return Math.max(0.05, Math.abs(mouthY - eyeY) * 2);
  }
  return 0.2;
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
