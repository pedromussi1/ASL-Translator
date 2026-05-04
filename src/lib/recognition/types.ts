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

export interface FrameResult {
  hands: DetectedHand[];
  timestampMs: number;
}

export interface NormalizedHand {
  vector: Float32Array;
  handedness: "Left" | "Right";
}

export interface Prediction {
  label: string;
  confidence: number;
}
