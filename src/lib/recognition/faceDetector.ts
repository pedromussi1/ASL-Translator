import type { DetectedFace, FaceKeypoint } from "./types";

const WASM_BASE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";

export interface FaceDetectorOptions {
  delegate?: "GPU" | "CPU";
  modelUrl?: string;
  wasmBaseUrl?: string;
}

/**
 * Browser-only wrapper around MediaPipe Tasks-Vision FaceDetector. We use
 * the cheap detector (bbox + 6 keypoints) rather than FaceLandmarker (478
 * points) because all we need is "where is the head" — eyes, nose tip,
 * mouth center, ear tragions are enough to anchor hand-relative motions.
 */
export class FaceDetectorEngine {
  private detector: import("@mediapipe/tasks-vision").FaceDetector | null = null;
  private lastTimestampMs = 0;

  private constructor() {}

  static async create(opts: FaceDetectorOptions = {}): Promise<FaceDetectorEngine> {
    if (typeof window === "undefined") {
      throw new Error("FaceDetectorEngine is browser-only.");
    }
    const { FilesetResolver, FaceDetector } = await import("@mediapipe/tasks-vision");
    const vision = await FilesetResolver.forVisionTasks(opts.wasmBaseUrl ?? WASM_BASE_URL);
    const engine = new FaceDetectorEngine();
    engine.detector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: opts.modelUrl ?? MODEL_URL,
        delegate: opts.delegate ?? "GPU",
      },
      runningMode: "VIDEO",
    });
    return engine;
  }

  /**
   * Detect the most prominent face in the current video frame. Returns null
   * if no face was found.
   */
  detect(
    video: HTMLVideoElement,
    timestampMs: number = performance.now(),
  ): DetectedFace | null {
    if (!this.detector) throw new Error("FaceDetector not initialized.");
    if (timestampMs <= this.lastTimestampMs) timestampMs = this.lastTimestampMs + 1;
    this.lastTimestampMs = timestampMs;

    const result = this.detector.detectForVideo(video, timestampMs);
    if (!result.detections || result.detections.length === 0) return null;

    // Prefer the largest face (most likely the user, not background bystanders).
    let best = result.detections[0];
    let bestScore = scoreOf(best);
    for (let i = 1; i < result.detections.length; i++) {
      const det = result.detections[i];
      const s = scoreOf(det);
      if (s > bestScore) {
        bestScore = s;
        best = det;
      }
    }

    const face: DetectedFace = {};
    for (const k of best.keypoints ?? []) {
      const label = (k as { label?: string }).label;
      const point: FaceKeypoint = { x: k.x, y: k.y };
      switch (label) {
        case "right_eye":
          face.rightEye = point;
          break;
        case "left_eye":
          face.leftEye = point;
          break;
        case "nose_tip":
          face.noseTip = point;
          break;
        case "mouth_center":
          face.mouthCenter = point;
          break;
        case "right_ear_tragion":
          face.rightEar = point;
          break;
        case "left_ear_tragion":
          face.leftEar = point;
          break;
      }
    }

    return face;
  }

  close(): void {
    this.detector?.close();
    this.detector = null;
  }
}

function scoreOf(det: { boundingBox?: { width: number; height: number } }): number {
  const bb = det.boundingBox;
  return bb ? bb.width * bb.height : 0;
}
