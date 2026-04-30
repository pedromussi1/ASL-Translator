"use client";

import type { DetectedFace, DetectedHand } from "@/lib/recognition/types";

const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],         // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],         // index
  [5, 9], [9, 10], [10, 11], [11, 12],    // middle
  [9, 13], [13, 14], [14, 15], [15, 16],  // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17],                                 // palm
];

export function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  hands: DetectedHand[],
  width: number,
  height: number,
  face?: DetectedFace | null,
): void {
  ctx.clearRect(0, 0, width, height);

  for (const hand of hands) {
    const color = hand.handedness === "Right" ? "#22d3ee" : "#f472b6";
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, width / 320);
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      const p = hand.landmarks[a];
      const q = hand.landmarks[b];
      ctx.moveTo(p.x * width, p.y * height);
      ctx.lineTo(q.x * width, q.y * height);
    }
    ctx.stroke();

    ctx.fillStyle = color;
    const r = Math.max(3, width / 240);
    for (const p of hand.landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (face) {
    const r = Math.max(4, width / 180);
    const points: Array<[{ x: number; y: number } | undefined, string]> = [
      [face.rightEye, "rEye"],
      [face.leftEye, "lEye"],
      [face.noseTip, "nose"],
      [face.mouthCenter, "mouth"],
      [face.rightEar, "rEar"],
      [face.leftEar, "lEar"],
    ];
    ctx.fillStyle = "#facc15"; // amber
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.lineWidth = 1;
    ctx.font = `${Math.max(10, width / 90)}px ui-sans-serif, system-ui`;
    for (const [p, label] of points) {
      if (!p) continue;
      const cx = p.x * width;
      const cy = p.y * height;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillText(label, cx + r + 2, cy + 3);
    }
  }
}
