import type { NormalizedHand, Point3, RawHandLandmarks } from "./types";

/**
 * Convert raw MediaPipe hand landmarks (21 points, normalized to [0,1] image
 * space) into a translation/scale/orientation-invariant feature vector suitable
 * for a static-pose classifier.
 *
 * Steps:
 *   1. Translate so the wrist (landmark 0) is at the origin.
 *   2. Scale so the distance from the wrist (0) to the middle-finger MCP (9)
 *      is 1. This gives us a hand-size-invariant representation.
 *   3. Optionally mirror across X so left/right hands map to the same feature
 *      space (the classifier doesn't have to learn handedness).
 *
 * Output is a Float32Array of length 63 (21 * 3) ordered [x0, y0, z0, x1, ...].
 */
export function normalizeHand(
  landmarks: RawHandLandmarks,
  handedness: "Left" | "Right",
  opts: { mirrorLeft?: boolean } = {},
): NormalizedHand {
  if (landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, got ${landmarks.length}.`);
  }
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];

  const dx = middleMcp.x - wrist.x;
  const dy = middleMcp.y - wrist.y;
  const dz = middleMcp.z - wrist.z;
  const scale = Math.hypot(dx, dy, dz) || 1e-6;

  const mirror = (opts.mirrorLeft ?? true) && handedness === "Left";

  const out = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    const p = landmarks[i];
    let x = (p.x - wrist.x) / scale;
    const y = (p.y - wrist.y) / scale;
    const z = (p.z - wrist.z) / scale;
    if (mirror) x = -x;
    out[i * 3] = x;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = z;
  }
  return { vector: out, handedness };
}

/**
 * Helper for tests: build a synthetic 21-point hand at given offset/scale.
 * Returns a flat A→T pose where each landmark is at (i, 0, 0) before transforms.
 */
export function syntheticHand(opts: {
  origin?: Point3;
  scale?: number;
} = {}): RawHandLandmarks {
  const origin = opts.origin ?? { x: 0.5, y: 0.5, z: 0 };
  const scale = opts.scale ?? 0.01;
  const pts: RawHandLandmarks = [];
  for (let i = 0; i < 21; i++) {
    pts.push({ x: origin.x + i * scale, y: origin.y, z: origin.z });
  }
  return pts;
}
