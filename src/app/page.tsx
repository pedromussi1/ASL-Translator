"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CalibrationOverlay,
  type RecordingPhase,
} from "@/components/CalibrationOverlay";
import { CameraView, type CameraStatus } from "@/components/CameraView";
import {
  CALIBRATION_FRAMES_PER_LETTER,
  CalibratedClassifier,
  LetterRecorder,
  type CalibrationData,
} from "@/lib/recognition/calibratedClassifier";
import { AlphabetClassifier } from "@/lib/recognition/classifier";
import { normalizeHand } from "@/lib/recognition/normalize";
import { PredictionSmoother } from "@/lib/recognition/smoother";
import type { DetectedHand, FrameResult, Prediction } from "@/lib/recognition/types";
import { WordBuffer } from "@/lib/recognition/wordBuffer";
import { WebSpeechProvider } from "@/lib/tts/webSpeech";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const PREVIEW_MS = 1200; // "Get ready…" before recording starts on Record click
const SAVED_FLASH_MS = 700; // how long to show "Saved ✓" before returning to grid

type Mode = "translate" | "calibrate";

type ClassifierState =
  | { kind: "loading" }
  | { kind: "trained"; cls: AlphabetClassifier }
  | { kind: "calibrated"; cls: CalibratedClassifier; sampleCounts: Record<string, number> }
  | { kind: "error"; message: string };

export default function Home() {
  const [started, setStarted] = useState(false);
  const [status, setStatus] = useState<CameraStatus>({ kind: "idle" });
  const [classifierState, setClassifierState] = useState<ClassifierState>({ kind: "loading" });
  const [transcript, setTranscript] = useState<string[]>([]);
  const [currentWord, setCurrentWord] = useState("");
  const [tentative, setTentative] = useState<{ letter: string | null; conf: number }>({
    letter: null,
    conf: 0,
  });

  const [mode, setMode] = useState<Mode>("translate");
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);
  const [recordingLetter, setRecordingLetter] = useState<string | null>(null);
  const [recordingPhase, setRecordingPhase] = useState<RecordingPhase>("idle");
  const [recordedCount, setRecordedCount] = useState(0);

  const smootherRef = useRef<PredictionSmoother | null>(null);
  const bufferRef = useRef<WordBuffer | null>(null);
  const ttsRef = useRef<WebSpeechProvider | null>(null);
  const inflightRef = useRef(false);
  const classifierRef = useRef<AlphabetClassifier | CalibratedClassifier | null>(null);
  const recorderRef = useRef<LetterRecorder | null>(null);

  // Refs that mirror state for the async onFrame closure.
  const modeRef = useRef<Mode>("translate");
  const recordingLetterRef = useRef<string | null>(null);
  const recordingPhaseRef = useRef<RecordingPhase>("idle");

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);
  useEffect(() => {
    recordingLetterRef.current = recordingLetter;
  }, [recordingLetter]);
  useEffect(() => {
    recordingPhaseRef.current = recordingPhase;
  }, [recordingPhase]);

  // Singletons.
  useEffect(() => {
    bufferRef.current = new WordBuffer();
    smootherRef.current = new PredictionSmoother(5);
    ttsRef.current = new WebSpeechProvider();
    const off = bufferRef.current.on((e) => {
      if (e.type === "letter_committed") {
        setCurrentWord(e.word);
      } else if (e.type === "word_committed") {
        setTranscript((prev) => [...prev, e.word]);
        setCurrentWord("");
        ttsRef.current?.speak(e.word);
      } else if (e.type === "tentative_change") {
        setTentative({ letter: e.letter, conf: e.confidence });
      } else if (e.type === "cleared") {
        setCurrentWord("");
        setTentative({ letter: null, conf: 0 });
      }
    });
    return () => {
      off();
      ttsRef.current?.cancel();
    };
  }, []);

  // Choose / load the active classifier when the camera starts.
  useEffect(() => {
    if (!started) return;
    let cancelled = false;
    setClassifierState({ kind: "loading" });

    const data = CalibratedClassifier.loadDataFromStorage();
    setCalibrationData(data);
    if (data && Object.keys(data.prototypes).length > 0) {
      const cls = CalibratedClassifier.fromData(data);
      classifierRef.current = cls;
      setClassifierState({ kind: "calibrated", cls, sampleCounts: data.sampleCounts });
      return;
    }

    AlphabetClassifier.load({
      modelUrl: "/models/asl_alphabet_v1/alphabet.onnx",
      labelsUrl: "/models/asl_alphabet_v1/labels.json",
    })
      .then((cls) => {
        if (cancelled) {
          cls.close();
          return;
        }
        classifierRef.current = cls;
        setClassifierState({ kind: "trained", cls });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setClassifierState({
            kind: "error",
            message: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [started]);

  // Recording phase machine: idle → (Record click) → preview → recording → saved → idle (back to grid).
  useEffect(() => {
    if (mode !== "calibrate" || recordingLetter === null) return;
    if (recordingPhase === "preview") {
      const t = setTimeout(() => {
        recorderRef.current = new LetterRecorder(recordingLetter);
        setRecordedCount(0);
        setRecordingPhase("recording");
      }, PREVIEW_MS);
      return () => clearTimeout(t);
    }
    if (recordingPhase === "saved") {
      const t = setTimeout(() => {
        setRecordingLetter(null);
        setRecordingPhase("idle");
      }, SAVED_FLASH_MS);
      return () => clearTimeout(t);
    }
  }, [mode, recordingPhase, recordingLetter]);

  const onFrame = useCallback(async (result: FrameResult) => {
    // CALIBRATION RECORDING
    if (modeRef.current === "calibrate") {
      if (recordingPhaseRef.current !== "recording") return;
      const letter = recordingLetterRef.current;
      const recorder = recorderRef.current;
      if (!letter || !recorder) return;
      if (result.hands.length === 0) return;
      const hand = pickPrimaryHand(result.hands);
      const norm = normalizeHand(hand.landmarks, hand.handedness);
      recorder.push(norm);
      const count = recorder.count;
      setRecordedCount(count);
      if (count >= CALIBRATION_FRAMES_PER_LETTER) {
        const proto = recorder.buildPrototype();
        if (proto) {
          const updated = CalibratedClassifier.upsertLetter(letter, proto, count);
          setCalibrationData(updated);
          const cls = CalibratedClassifier.fromData(updated);
          classifierRef.current = cls;
          setClassifierState({
            kind: "calibrated",
            cls,
            sampleCounts: updated.sampleCounts,
          });
        }
        recorderRef.current = null;
        setRecordingPhase("saved");
      }
      return;
    }

    // TRANSLATION
    const buf = bufferRef.current;
    const smoother = smootherRef.current;
    if (!buf || !smoother) return;
    const cls = classifierRef.current;
    if (!cls || result.hands.length === 0) {
      buf.feed(smoother.push(null), result.timestampMs);
      return;
    }
    if (inflightRef.current) return;
    inflightRef.current = true;
    try {
      const hand = pickPrimaryHand(result.hands);
      const norm = normalizeHand(hand.landmarks, hand.handedness);
      const raw: Prediction = await cls.recognize(norm);
      buf.feed(smoother.push(raw), result.timestampMs);
    } finally {
      inflightRef.current = false;
    }
  }, []);

  const enterCalibration = () => {
    bufferRef.current?.clear();
    smootherRef.current?.reset();
    ttsRef.current?.cancel();
    setMode("calibrate");
    setRecordingLetter(null);
    setRecordingPhase("idle");
  };

  const exitCalibration = () => {
    setMode("translate");
    setRecordingLetter(null);
    setRecordingPhase("idle");
    recorderRef.current = null;
  };

  const pickLetterToRecord = (letter: string) => {
    setRecordingLetter(letter);
    setRecordingPhase("idle");
    setRecordedCount(0);
  };

  const startRecordingClicked = () => {
    setRecordingPhase("preview");
  };

  const cancelRecording = () => {
    recorderRef.current = null;
    setRecordingLetter(null);
    setRecordingPhase("idle");
    setRecordedCount(0);
  };

  const clearLetter = (letter: string) => {
    const updated = CalibratedClassifier.removeLetter(letter);
    setCalibrationData(updated);
    if (updated && Object.keys(updated.prototypes).length > 0) {
      const cls = CalibratedClassifier.fromData(updated);
      classifierRef.current = cls;
      setClassifierState({
        kind: "calibrated",
        cls,
        sampleCounts: updated.sampleCounts,
      });
    } else {
      // No more prototypes → fall back to trained model on next camera start.
      CalibratedClassifier.clearStorage();
      setCalibrationData(null);
      setStarted(false);
      setTimeout(() => setStarted(true), 0);
    }
  };

  const clearAllCalibration = () => {
    CalibratedClassifier.clearStorage();
    setCalibrationData(null);
    setStarted(false);
    setTimeout(() => setStarted(true), 0);
  };

  const handleClear = () => {
    bufferRef.current?.clear();
    smootherRef.current?.reset();
    setTranscript([]);
    ttsRef.current?.cancel();
  };

  const handleSpace = () => {
    bufferRef.current?.feed(null, performance.now() + 10_000);
  };

  const calibratedLetters = new Set(Object.keys(calibrationData?.prototypes ?? {}));
  const sampleCounts = calibrationData?.sampleCounts ?? {};

  return (
    <main className="flex flex-1 w-full mx-auto max-w-5xl flex-col gap-6 p-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">ASL Translator</h1>
        <span className="text-xs uppercase tracking-widest text-zinc-400">
          v0.1 · alphabet
        </span>
      </header>

      <p className="text-sm text-zinc-400 max-w-2xl">
        Phase 1: fingerspelling alphabet. Hold each letter steady for ~300ms;
        words commit when you drop your hand. Everything runs in your browser
        — no frames leave your device.
      </p>

      <section className="rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800 shadow-lg relative">
        {started ? (
          <>
            <CameraView onStatusChange={setStatus} onFrame={onFrame} className="aspect-video" />
            {mode === "calibrate" && (
              <CalibrationOverlay
                letters={ALPHABET}
                calibratedLetters={calibratedLetters}
                sampleCounts={sampleCounts}
                recordingLetter={recordingLetter}
                recordingPhase={recordingPhase}
                recordedCount={recordedCount}
                framesPerLetter={CALIBRATION_FRAMES_PER_LETTER}
                onPickLetter={pickLetterToRecord}
                onStartRecording={startRecordingClicked}
                onCancelRecording={cancelRecording}
                onClearLetter={clearLetter}
                onClose={exitCalibration}
              />
            )}
          </>
        ) : (
          <div className="aspect-video flex items-center justify-center">
            <button
              type="button"
              onClick={() => setStarted(true)}
              className="rounded-full bg-cyan-500 px-6 py-3 text-sm font-medium text-zinc-950 hover:bg-cyan-400 transition"
            >
              Start camera
            </button>
          </div>
        )}
      </section>

      <ClassifierBanner
        state={classifierState}
        started={started}
        mode={mode}
        onCalibrate={enterCalibration}
        onClearCalibration={clearAllCalibration}
      />

      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Stat label="Detecting">
          <span
            className={tentative.letter ? "text-cyan-300" : "text-zinc-500"}
            style={{ fontSize: "2.25rem", lineHeight: 1, fontWeight: 600 }}
          >
            {tentative.letter ?? "—"}
          </span>
          <ConfidenceBar value={tentative.conf} />
        </Stat>

        <Stat label="Word">
          <span className="text-2xl font-mono text-zinc-100 min-h-8 inline-block">
            {currentWord || <span className="text-zinc-600">…</span>}
          </span>
        </Stat>

        <Stat label="Transcript">
          <span className="text-sm font-mono text-zinc-300 leading-relaxed break-words">
            {transcript.length > 0 ? (
              transcript.join(" ")
            ) : (
              <span className="text-zinc-600">nothing yet</span>
            )}
          </span>
        </Stat>
      </section>

      <div className="flex gap-2 text-sm">
        <button
          type="button"
          onClick={handleSpace}
          className="px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 transition"
        >
          End word
        </button>
        <button
          type="button"
          onClick={handleClear}
          className="px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 transition"
        >
          Clear
        </button>
      </div>

      <StatusBar status={status} />

      <footer className="text-xs text-zinc-500 pt-4 border-t border-zinc-800">
        Personal learning project. Built on MediaPipe Tasks + ONNX Runtime Web.
      </footer>
    </main>
  );
}

function pickPrimaryHand(hands: DetectedHand[]): DetectedHand {
  let best = hands[0];
  let bestArea = 0;
  for (const h of hands) {
    let xMin = 1, yMin = 1, xMax = 0, yMax = 0;
    for (const p of h.landmarks) {
      if (p.x < xMin) xMin = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.x > xMax) xMax = p.x;
      if (p.y > yMax) yMax = p.y;
    }
    const area = Math.max(0, xMax - xMin) * Math.max(0, yMax - yMin);
    if (area > bestArea) {
      bestArea = area;
      best = h;
    }
  }
  return best;
}

function ClassifierBanner({
  state,
  started,
  mode,
  onCalibrate,
  onClearCalibration,
}: {
  state: ClassifierState;
  started: boolean;
  mode: Mode;
  onCalibrate: () => void;
  onClearCalibration: () => void;
}) {
  if (!started) return null;
  if (mode === "calibrate") {
    return (
      <div className="text-xs text-cyan-300">
        Calibration mode — pick any letter to record or recalibrate it.
      </div>
    );
  }
  if (state.kind === "loading") {
    return <div className="text-xs text-amber-300">Loading classifier…</div>;
  }
  if (state.kind === "error") {
    return <div className="text-xs text-rose-400">Classifier failed: {state.message}</div>;
  }
  if (state.kind === "calibrated") {
    const letterCount = Object.keys(state.sampleCounts).length;
    return (
      <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
        <span className="text-emerald-300">
          Using your personal calibration ({letterCount} letter{letterCount === 1 ? "" : "s"}).
        </span>
        <button
          type="button"
          onClick={onCalibrate}
          className="text-cyan-300 hover:text-cyan-200 underline-offset-2 hover:underline"
        >
          Open calibration
        </button>
        <button
          type="button"
          onClick={onClearCalibration}
          className="text-zinc-500 hover:text-zinc-300 underline-offset-2 hover:underline"
        >
          Clear all
        </button>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
      <span>Using the trained model. Accuracy varies by hand & lighting.</span>
      <button
        type="button"
        onClick={onCalibrate}
        className="text-cyan-300 hover:text-cyan-200 underline-offset-2 hover:underline"
      >
        Calibrate to your hand
      </button>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-4 flex flex-col gap-2 min-h-24">
      <span className="text-xs uppercase tracking-widest text-zinc-500">{label}</span>
      <div>{children}</div>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="h-1 rounded bg-zinc-800 overflow-hidden mt-2">
      <div
        className={pct >= 70 ? "h-full bg-emerald-500" : "h-full bg-amber-500"}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function StatusBar({ status }: { status: CameraStatus }) {
  if (status.kind === "idle") {
    return <div className="text-xs text-zinc-500">Camera idle.</div>;
  }
  if (status.kind === "starting") {
    return <div className="text-xs text-zinc-400">Initializing…</div>;
  }
  if (status.kind === "error") {
    return <div className="text-xs text-rose-400">Error: {status.error.message}</div>;
  }
  const dim = status.brightness < 60;
  return (
    <div className="flex flex-wrap gap-4 text-xs text-zinc-400">
      <span>
        Brightness:{" "}
        <span className={dim ? "text-amber-400" : "text-emerald-400"}>
          {status.brightness.toFixed(0)}
          {dim ? " (low light)" : ""}
        </span>
      </span>
      <span>
        Hands:{" "}
        <span className={status.handsDetected > 0 ? "text-emerald-400" : "text-zinc-500"}>
          {status.handsDetected}
        </span>
      </span>
    </div>
  );
}
