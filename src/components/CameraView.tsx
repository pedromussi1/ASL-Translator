"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CameraStream, estimateBrightness, type CameraError } from "@/lib/camera";
import { FaceDetectorEngine } from "@/lib/recognition/faceDetector";
import { HandLandmarkerEngine } from "@/lib/recognition/handLandmarker";
import type { DetectedFace, FrameResult } from "@/lib/recognition/types";
import { drawLandmarks } from "./LandmarkOverlay";

export type CameraStatus =
  | { kind: "idle" }
  | { kind: "starting" }
  | { kind: "running"; brightness: number; handsDetected: number }
  | { kind: "error"; error: CameraError };

export interface CameraViewProps {
  onFrame?: (result: FrameResult) => void;
  onStatusChange?: (status: CameraStatus) => void;
  showLandmarks?: boolean;
  className?: string;
}

export function CameraView({
  onFrame,
  onStatusChange,
  showLandmarks = true,
  className,
}: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lumaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cameraRef = useRef<CameraStream | null>(null);
  const engineRef = useRef<HandLandmarkerEngine | null>(null);
  const faceRef = useRef<FaceDetectorEngine | null>(null);
  const rafRef = useRef<number | null>(null);
  const lumaTickRef = useRef<number>(0);

  const [status, setStatus] = useState<CameraStatus>({ kind: "idle" });

  const updateStatus = useCallback(
    (next: CameraStatus) => {
      setStatus(next);
      onStatusChange?.(next);
    },
    [onStatusChange],
  );

  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    updateStatus({ kind: "starting" });

    (async () => {
      try {
        const cam = await CameraStream.start(video, {
          width: 1280,
          height: 720,
          frameRate: 30,
        });
        if (cancelled) {
          cam.stop();
          return;
        }
        cameraRef.current = cam;

        const engine = await HandLandmarkerEngine.create({
          numHands: 2,
          delegate: "GPU",
        }).catch(async () =>
          // Fall back to CPU if GPU init fails (older webview, etc.).
          HandLandmarkerEngine.create({ numHands: 2, delegate: "CPU" }),
        );
        if (cancelled) {
          engine.close();
          cam.stop();
          return;
        }
        engineRef.current = engine;

        // Face detector is best-effort: if it fails to load (older webview,
        // CDN hiccup, etc.), we just keep going without face context.
        FaceDetectorEngine.create({ delegate: "GPU" })
          .catch(() => FaceDetectorEngine.create({ delegate: "CPU" }))
          .then((face) => {
            if (cancelled) {
              face.close();
              return;
            }
            faceRef.current = face;
          })
          .catch(() => {
            /* ignore — dynamic matchers fall back gracefully */
          });

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        const lumaCanvas = lumaCanvasRef.current ?? document.createElement("canvas");
        lumaCanvas.width = 64;
        lumaCanvas.height = 36;
        lumaCanvasRef.current = lumaCanvas;
        const lumaCtx = lumaCanvas.getContext("2d", { willReadFrequently: true });

        let lastBrightness = 128;
        let lastHandCount = 0;

        // Face detection is throttled — running every frame doubles inference
        // cost and the head doesn't move much between frames anyway. Re-detect
        // every ~3 frames (~100ms at 30fps) and reuse the last result in between.
        let lastFace: DetectedFace | null = null;
        let faceTick = 0;

        const loop = () => {
          if (cancelled || !ctx || !engineRef.current) return;
          const t = performance.now();

          if (video.readyState >= 2) {
            const handResult = engineRef.current.detect(video, t);
            lastHandCount = handResult.hands.length;
            if (showLandmarks) {
              drawLandmarks(ctx, handResult.hands, canvas.width, canvas.height);
            }

            faceTick = (faceTick + 1) % 3;
            if (faceTick === 0 && faceRef.current) {
              try {
                lastFace = faceRef.current.detect(video, t);
              } catch {
                /* ignore transient detection errors */
              }
            }

            onFrame?.({ ...handResult, face: lastFace });
          }

          // Sample brightness ~ every 500ms.
          if (lumaCtx && t - lumaTickRef.current > 500) {
            lumaTickRef.current = t;
            try {
              lastBrightness = estimateBrightness(video, lumaCtx);
              updateStatus({
                kind: "running",
                brightness: lastBrightness,
                handsDetected: lastHandCount,
              });
            } catch {
              /* ignore transient draw errors */
            }
          }

          rafRef.current = requestAnimationFrame(loop);
        };

        updateStatus({
          kind: "running",
          brightness: lastBrightness,
          handsDetected: 0,
        });
        rafRef.current = requestAnimationFrame(loop);
      } catch (err) {
        if (!cancelled) {
          const error = (err as CameraError)?.kind
            ? (err as CameraError)
            : ({ kind: "other", message: String(err) } satisfies CameraError);
          updateStatus({ kind: "error", error });
        }
      }
    })();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      engineRef.current?.close();
      engineRef.current = null;
      faceRef.current?.close();
      faceRef.current = null;
      cameraRef.current?.stop();
      cameraRef.current = null;
    };
    // We intentionally run this effect once per mount; consumer-supplied callbacks
    // are stored in refs by React under the hood when wrapped in useCallback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={className} style={{ position: "relative", width: "100%" }}>
      <video
        ref={videoRef}
        playsInline
        muted
        style={{
          width: "100%",
          height: "auto",
          display: "block",
          transform: "scaleX(-1)", // mirror like a selfie
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          transform: "scaleX(-1)",
        }}
      />
      {status.kind === "starting" && (
        <div style={overlayStyle}>Starting camera…</div>
      )}
      {status.kind === "error" && (
        <div style={{ ...overlayStyle, background: "rgba(127,29,29,0.85)" }}>
          {status.error.message}
        </div>
      )}
    </div>
  );
}

const overlayStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "white",
  background: "rgba(0,0,0,0.5)",
  fontSize: "1.25rem",
};
