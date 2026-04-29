"use client";

export type CalibrationPhase = "preview" | "recording" | "complete";

export interface CalibrationOverlayProps {
  letters: readonly string[];
  index: number;
  phase: CalibrationPhase;
  recordedCount: number;
  framesPerLetter: number;
  recordedLetters: number;
  onSkip: () => void;
  onRetry: () => void;
  onCancel: () => void;
  onFinish: () => void;
}

export function CalibrationOverlay({
  letters,
  index,
  phase,
  recordedCount,
  framesPerLetter,
  recordedLetters,
  onSkip,
  onRetry,
  onCancel,
  onFinish,
}: CalibrationOverlayProps) {
  const letter = letters[index];
  const total = letters.length;
  const overallProgress = Math.round(((index + (phase === "complete" ? 1 : 0)) / total) * 100);
  const recordingProgress =
    phase === "recording" ? Math.min(1, recordedCount / framesPerLetter) : 0;

  if (phase === "complete") {
    return (
      <div className="absolute inset-0 bg-zinc-950/95 flex flex-col items-center justify-center gap-6 p-6 text-center">
        <span className="text-4xl">✓</span>
        <h2 className="text-2xl font-semibold">Calibration complete</h2>
        <p className="text-sm text-zinc-400 max-w-md">
          Captured {recordedLetters} of {total} letters. Confidence will be much
          higher for letters you calibrated, low for any you skipped — those
          will fall back to "—".
        </p>
        <button
          type="button"
          onClick={onFinish}
          className="rounded-full bg-emerald-500 px-6 py-3 text-sm font-medium text-zinc-950 hover:bg-emerald-400 transition"
        >
          Start translating
        </button>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 bg-zinc-950/85 flex flex-col items-stretch justify-between p-6 backdrop-blur-sm">
      <div className="flex items-baseline justify-between text-xs uppercase tracking-widest text-zinc-400">
        <span>
          Calibrating · letter {index + 1} of {total}
        </span>
        <button
          type="button"
          onClick={onCancel}
          className="text-zinc-400 hover:text-zinc-100"
        >
          Cancel
        </button>
      </div>

      <div className="flex flex-col items-center gap-3">
        <div
          className="font-bold leading-none text-cyan-300"
          style={{ fontSize: "min(40vh, 18rem)" }}
        >
          {letter}
        </div>
        <div
          className={
            "text-lg uppercase tracking-widest " +
            (phase === "recording" ? "text-emerald-300" : "text-amber-300")
          }
        >
          {phase === "recording" ? "Hold steady" : "Get ready…"}
        </div>
        <p className="text-sm text-zinc-400 max-w-md text-center">
          Sign the letter <span className="text-zinc-200 font-mono">{letter}</span> in front of the camera.
          {phase === "preview"
            ? " Recording will start in a moment."
            : " Slight motion is fine — keep your hand visible."}
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <div>
          <div className="flex justify-between text-[10px] uppercase tracking-widest text-zinc-500 mb-1">
            <span>Letter capture</span>
            <span>
              {Math.round(recordingProgress * 100)}% (
              {recordedCount}/{framesPerLetter})
            </span>
          </div>
          <div className="h-1.5 rounded bg-zinc-800 overflow-hidden">
            <div
              className="h-full bg-emerald-500 transition-[width] duration-100"
              style={{ width: `${recordingProgress * 100}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-[10px] uppercase tracking-widest text-zinc-500 mb-1">
            <span>Overall</span>
            <span>{overallProgress}%</span>
          </div>
          <div className="h-1.5 rounded bg-zinc-800 overflow-hidden">
            <div
              className="h-full bg-cyan-500 transition-[width] duration-200"
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>
        <div className="flex gap-2 justify-end text-xs">
          <button
            type="button"
            onClick={onRetry}
            className="px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 transition"
          >
            Retry letter
          </button>
          <button
            type="button"
            onClick={onSkip}
            className="px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 transition"
          >
            Skip letter
          </button>
        </div>
      </div>
    </div>
  );
}
