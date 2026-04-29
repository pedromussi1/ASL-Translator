"use client";

export type RecordingPhase = "idle" | "preview" | "recording" | "saved";

export interface CalibrationOverlayProps {
  letters: readonly string[];
  /** Letters that already have a saved prototype. */
  calibratedLetters: ReadonlySet<string>;
  sampleCounts: Readonly<Record<string, number>>;

  /** When set, we're inside the recording view for that letter; else we show the grid. */
  recordingLetter: string | null;
  recordingPhase: RecordingPhase;
  recordedCount: number;
  framesPerLetter: number;

  onPickLetter: (letter: string) => void;
  onStartRecording: () => void;
  onCancelRecording: () => void;
  onClearLetter: (letter: string) => void;
  onClose: () => void;
}

export function CalibrationOverlay(props: CalibrationOverlayProps) {
  if (props.recordingLetter !== null) {
    return <RecordingView {...props} />;
  }
  return <GridView {...props} />;
}

function GridView({
  letters,
  calibratedLetters,
  sampleCounts,
  onPickLetter,
  onClearLetter,
  onClose,
}: CalibrationOverlayProps) {
  const calibratedCount = calibratedLetters.size;
  return (
    <div className="absolute inset-0 bg-zinc-950/95 flex flex-col p-6 gap-5 backdrop-blur-sm overflow-y-auto">
      <header className="flex items-baseline justify-between">
        <div>
          <h2 className="text-xl font-semibold">Calibrate to your hand</h2>
          <p className="text-xs text-zinc-400 mt-1">
            Tap a letter, sign it, and press Record. {calibratedCount}/{letters.length} done.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-xs uppercase tracking-widest text-zinc-400 hover:text-zinc-100"
        >
          Done
        </button>
      </header>

      <div className="grid grid-cols-5 sm:grid-cols-7 gap-2">
        {letters.map((letter) => {
          const done = calibratedLetters.has(letter);
          return (
            <div key={letter} className="relative group">
              <button
                type="button"
                onClick={() => onPickLetter(letter)}
                className={
                  "w-full aspect-square rounded-lg border text-2xl font-semibold transition flex flex-col items-center justify-center gap-0.5 " +
                  (done
                    ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/20"
                    : "bg-zinc-900 border-zinc-800 text-zinc-300 hover:bg-zinc-800")
                }
                title={
                  done
                    ? `Recalibrate ${letter} (${sampleCounts[letter] ?? 0} samples)`
                    : `Calibrate ${letter}`
                }
              >
                <span>{letter}</span>
                <span className="text-[9px] uppercase tracking-widest text-current opacity-70">
                  {done ? `${sampleCounts[letter] ?? 0} samples` : "not calibrated"}
                </span>
              </button>
              {done && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClearLetter(letter);
                  }}
                  className="absolute top-1 right-1 hidden group-hover:block text-zinc-400 hover:text-rose-400 text-xs px-1"
                  title={`Clear ${letter}`}
                >
                  ✕
                </button>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-xs text-zinc-500 pt-2 border-t border-zinc-800">
        Tip: skip <span className="font-mono text-zinc-300">J</span> and{" "}
        <span className="font-mono text-zinc-300">Z</span> — they need motion. Hover a calibrated letter for a clear (✕) shortcut.
      </p>
    </div>
  );
}

function RecordingView({
  recordingLetter,
  recordingPhase,
  recordedCount,
  framesPerLetter,
  onStartRecording,
  onCancelRecording,
}: CalibrationOverlayProps) {
  const letter = recordingLetter!;
  const progress =
    recordingPhase === "recording" ? Math.min(1, recordedCount / framesPerLetter) : 0;

  return (
    <div className="absolute inset-0 bg-zinc-950/85 flex flex-col p-6 backdrop-blur-sm">
      <header className="flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-widest text-zinc-400">
          Calibrating · {letter}
        </span>
        <button
          type="button"
          onClick={onCancelRecording}
          className="text-xs uppercase tracking-widest text-zinc-400 hover:text-zinc-100"
        >
          Back
        </button>
      </header>

      <div className="flex-1 flex flex-col items-center justify-center gap-3">
        <div
          className="font-bold leading-none text-cyan-300"
          style={{ fontSize: "min(38vh, 16rem)" }}
        >
          {letter}
        </div>
        <div
          className={
            "text-lg uppercase tracking-widest " +
            (recordingPhase === "recording"
              ? "text-emerald-300"
              : recordingPhase === "saved"
                ? "text-emerald-400"
                : recordingPhase === "preview"
                  ? "text-amber-300"
                  : "text-zinc-300")
          }
        >
          {recordingPhase === "recording"
            ? "Hold steady"
            : recordingPhase === "saved"
              ? "Saved ✓"
              : recordingPhase === "preview"
                ? "Get ready…"
                : "Sign the letter, then press Record"}
        </div>
      </div>

      <div className="flex flex-col gap-3">
        {recordingPhase === "recording" && (
          <div>
            <div className="flex justify-between text-[10px] uppercase tracking-widest text-zinc-500 mb-1">
              <span>Capture</span>
              <span>
                {Math.round(progress * 100)}% ({recordedCount}/{framesPerLetter})
              </span>
            </div>
            <div className="h-1.5 rounded bg-zinc-800 overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-[width] duration-100"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>
        )}
        <div className="flex gap-2 justify-end">
          {recordingPhase === "idle" || recordingPhase === "saved" ? (
            <button
              type="button"
              onClick={onStartRecording}
              className="rounded-full bg-cyan-500 px-6 py-3 text-sm font-medium text-zinc-950 hover:bg-cyan-400 transition"
            >
              Record {letter}
            </button>
          ) : (
            <button
              type="button"
              onClick={onCancelRecording}
              className="px-4 py-2 rounded bg-zinc-800 text-sm hover:bg-zinc-700 transition"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
