export interface CameraOptions {
  width?: number;
  height?: number;
  frameRate?: number;
  facingMode?: "user" | "environment";
  deviceId?: string;
}

export type CameraError =
  | { kind: "permission_denied"; message: string }
  | { kind: "no_device"; message: string }
  | { kind: "in_use"; message: string }
  | { kind: "unsupported"; message: string }
  | { kind: "other"; message: string };

export class CameraStream {
  private constructor(
    private readonly mediaStream: MediaStream,
    private readonly video: HTMLVideoElement,
  ) {}

  static async start(
    video: HTMLVideoElement,
    opts: CameraOptions = {},
  ): Promise<CameraStream> {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      throw {
        kind: "unsupported",
        message: "getUserMedia is not available in this environment.",
      } satisfies CameraError;
    }

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: {
        width: { ideal: opts.width ?? 1280 },
        height: { ideal: opts.height ?? 720 },
        frameRate: { ideal: opts.frameRate ?? 30 },
        facingMode: opts.facingMode ?? "user",
        ...(opts.deviceId ? { deviceId: { exact: opts.deviceId } } : {}),
      },
    };

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      throw mapMediaError(err);
    }

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await new Promise<void>((resolve, reject) => {
      const onLoaded = () => {
        video.removeEventListener("loadedmetadata", onLoaded);
        video.removeEventListener("error", onError);
        resolve();
      };
      const onError = () => {
        video.removeEventListener("loadedmetadata", onLoaded);
        video.removeEventListener("error", onError);
        reject({ kind: "other", message: "Video element error." } satisfies CameraError);
      };
      video.addEventListener("loadedmetadata", onLoaded);
      video.addEventListener("error", onError);
    });

    await video.play().catch(() => {
      // Autoplay may be blocked until user interaction; the caller is expected to retry.
    });

    return new CameraStream(stream, video);
  }

  stop(): void {
    for (const track of this.mediaStream.getTracks()) track.stop();
    if (this.video.srcObject === this.mediaStream) this.video.srcObject = null;
  }

  get stream(): MediaStream {
    return this.mediaStream;
  }

  get videoElement(): HTMLVideoElement {
    return this.video;
  }

  getSettings(): MediaTrackSettings | null {
    const track = this.mediaStream.getVideoTracks()[0];
    return track ? track.getSettings() : null;
  }
}

export async function listVideoDevices(): Promise<MediaDeviceInfo[]> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "videoinput");
}

/**
 * Average luma over a small downsampled grid. Returns a value in [0, 255].
 * Used for "is the room bright enough" readiness gate.
 */
export function estimateBrightness(
  video: HTMLVideoElement,
  ctx: CanvasRenderingContext2D,
): number {
  const w = 64;
  const h = Math.max(1, Math.round((video.videoHeight / video.videoWidth) * w));
  ctx.drawImage(video, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) {
    // Rec. 601 luma
    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return sum / (data.length / 4);
}

function mapMediaError(err: unknown): CameraError {
  if (err && typeof err === "object" && "name" in err) {
    const name = (err as { name: string }).name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      return { kind: "permission_denied", message: "Camera permission denied." };
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return { kind: "no_device", message: "No matching camera was found." };
    }
    if (name === "NotReadableError") {
      return { kind: "in_use", message: "Camera is in use by another app." };
    }
  }
  const message = err instanceof Error ? err.message : "Failed to start camera.";
  return { kind: "other", message };
}
