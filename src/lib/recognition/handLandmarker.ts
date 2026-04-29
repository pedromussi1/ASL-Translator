import type { DetectedHand, FrameResult, RawHandLandmarks } from "./types";

const WASM_BASE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

export interface HandLandmarkerOptions {
  numHands?: number;
  delegate?: "GPU" | "CPU";
  modelUrl?: string;
  wasmBaseUrl?: string;
}

/**
 * Browser-only wrapper around MediaPipe Tasks-Vision HandLandmarker.
 * Phase 1: loads model + WASM from CDN. Replace with self-hosted assets when commercial.
 */
export class HandLandmarkerEngine {
  private landmarker: import("@mediapipe/tasks-vision").HandLandmarker | null = null;
  private lastTimestampMs = 0;

  private constructor() {}

  static async create(opts: HandLandmarkerOptions = {}): Promise<HandLandmarkerEngine> {
    if (typeof window === "undefined") {
      throw new Error("HandLandmarkerEngine is browser-only.");
    }
    const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
    const vision = await FilesetResolver.forVisionTasks(opts.wasmBaseUrl ?? WASM_BASE_URL);
    const engine = new HandLandmarkerEngine();
    engine.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: opts.modelUrl ?? MODEL_URL,
        delegate: opts.delegate ?? "GPU",
      },
      numHands: opts.numHands ?? 2,
      runningMode: "VIDEO",
    });
    return engine;
  }

  /**
   * Detect hands in the current video frame. The video element must be playing
   * (readyState >= 2) before calling.
   */
  detect(video: HTMLVideoElement, timestampMs: number = performance.now()): FrameResult {
    if (!this.landmarker) throw new Error("HandLandmarker not initialized.");
    // MediaPipe requires monotonically increasing timestamps.
    if (timestampMs <= this.lastTimestampMs) timestampMs = this.lastTimestampMs + 1;
    this.lastTimestampMs = timestampMs;

    const result = this.landmarker.detectForVideo(video, timestampMs);

    const hands: DetectedHand[] = [];
    const len = result.landmarks?.length ?? 0;
    for (let i = 0; i < len; i++) {
      const landmarks = result.landmarks[i] as RawHandLandmarks;
      const worldLandmarks = (result.worldLandmarks?.[i] ?? landmarks) as RawHandLandmarks;
      const handednessCategory = result.handednesses?.[i]?.[0];
      const handedness =
        handednessCategory?.categoryName === "Left" ? "Left" : "Right";
      hands.push({
        landmarks,
        worldLandmarks,
        handedness,
        handednessScore: handednessCategory?.score ?? 0,
      });
    }
    return { hands, timestampMs };
  }

  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
