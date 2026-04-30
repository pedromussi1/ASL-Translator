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

    // BlazeFace short-range returns 6 keypoints in a fixed order; the JS SDK
    // doesn't populate the `label` strings, so we index by position. Order is
    // documented in MediaPipe's BlazeFace solution and is stable across
    // releases.
    //   0: right eye, 1: left eye, 2: nose tip, 3: mouth center,
    //   4: right ear tragion, 5: left ear tragion
    const kps = best.keypoints ?? [];
    const at = (i: number): FaceKeypoint | undefined => {
      const k = kps[i];
      return k ? { x: k.x, y: k.y } : undefined;
    };

    return {
      rightEye: at(0),
      leftEye: at(1),
      noseTip: at(2),
      mouthCenter: at(3),
      rightEar: at(4),
      leftEar: at(5),
    };
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
