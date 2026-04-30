export interface Point3 {
  x: number;
  y: number;
  z: number;
}

export type RawHandLandmarks = Point3[];

export interface DetectedHand {
  landmarks: RawHandLandmarks;
  worldLandmarks: RawHandLandmarks;
  handedness: "Left" | "Right";
  handednessScore: number;
}

export interface FaceKeypoint {
  x: number;
  y: number;
}

/**
 * Most-prominent face this frame, with its 6 named keypoints (all coords in
 * normalized [0, 1] image space matching the hand landmarks). Used by the
 * dynamic-sign matchers to anchor hand position to the user's actual head
 * rather than absolute screen coords.
 */
export interface DetectedFace {
  rightEye?: FaceKeypoint;
  leftEye?: FaceKeypoint;
  noseTip?: FaceKeypoint;
  mouthCenter?: FaceKeypoint;
  rightEar?: FaceKeypoint;
  leftEar?: FaceKeypoint;
}

/** Output of a single HandLandmarker run. */
export interface HandsResult {
  hands: DetectedHand[];
  timestampMs: number;
}

/** Combined per-frame signal handed to consumers (hands + face). */
export interface FrameResult extends HandsResult {
  /** Most prominent face if any was detected this frame, otherwise null. */
  face: DetectedFace | null;
}

export interface NormalizedHand {
  vector: Float32Array;
  handedness: "Left" | "Right";
}

export interface Prediction {
  label: string;
  confidence: number;
}
