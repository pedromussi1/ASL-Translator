"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CalibrationOverlay, type CalibrationPhase } from "@/components/CalibrationOverlay";
import { CameraView, type CameraStatus } from "@/components/CameraView";
import {
  CALIBRATION_FRAMES_PER_LETTER,
  CalibratedClassifier,
  CalibrationSession,
} from "@/lib/recognition/calibratedClassifier";
import { AlphabetClassifier } from "@/lib/recognition/classifier";
import { normalizeHand } from "@/lib/recognition/normalize";
import { PredictionSmoother } from "@/lib/recognition/smoother";
import type { DetectedHand, FrameResult, Prediction } from "@/lib/recognition/types";
import { WordBuffer } from "@/lib/recognition/wordBuffer";
import { WebSpeechProvider } from "@/lib/tts/webSpeech";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

type Mode = "translate" | "calibrate";

type ClassifierState =
  | { kind: "loading" }
  | { kind: "trained"; cls: AlphabetClassifier; calibrated: false }
  | { kind: "calibrated"; cls: CalibratedClassifier; sampleCounts: Record<string, number> }
  | { kind: "error"; message: string };

const PREVIEW_MS = 1500; // "Get ready…" duration before recording starts

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
  const [calIdx, setCalIdx] = useState(0);
  const [calPhase, setCalPhase] = useState<CalibrationPhase>("preview");
  const [calRecordedCount, setCalRecordedCount] = useState(0);

  const smootherRef = useRef<PredictionSmoother | null>(null);
  const bufferRef = useRef<WordBuffer | null>(null);
  const ttsRef = useRef<WebSpeechProvider | null>(null);
  const inflightRef = useRef(false);
  const classifierRef = useRef<AlphabetClassifier | CalibratedClassifier | null>(null);

  const calSessionRef = useRef<CalibrationSession | null>(null);
  // Snapshot of mode/phase/index for the async onFrame closure.
  const modeRef = useRef<Mode>("translate");
  const calIdxRef = useRef(0);
  const calPhaseRef = useRef<CalibrationPhase>("preview");

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);
  useEffect(() => {
    calIdxRef.current = calIdx;
  }, [calIdx]);
  useEffect(() => {
    calPhaseRef.current = calPhase;
  }, [calPhase]);

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

    const calibrated = CalibratedClassifier.loadFromStorage();
    if (calibrated) {
      const raw = localStorage.getItem("asl-translator:calibration:v1");
      let counts: Record<string, number> = {};
      try {
        const parsed = raw ? JSON.parse(raw) : null;
        counts = parsed?.sampleCounts ?? {};
      } catch {
        /* ignore */
      }
      classifierRef.current = calibrated;
      setClassifierState({ kind: "calibrated", cls: calibrated, sampleCounts: counts });
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
        setClassifierState({ kind: "trained", cls, calibrated: false });
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

  // Calibration: schedule the preview→recording transition.
  useEffect(() => {
    if (mode !== "calibrate" || calPhase !== "preview") return;
    const t = setTimeout(() => {
      setCalRecordedCount(0);
      setCalPhase("recording");
    }, PREVIEW_MS);
    return () => clearTimeout(t);
  }, [mode, calPhase, calIdx]);

  const advanceCalibration = useCallback(() => {
    if (calIdxRef.current + 1 >= ALPHABET.length) {
      setCalPhase("complete");
      return;
    }
    setCalIdx((i) => i + 1);
    setCalPhase("preview");
    setCalRecordedCount(0);
  }, []);

  const onFrame = useCallback(async (result: FrameResult) => {
    // CALIBRATION PATH
    if (modeRef.current === "calibrate") {
      if (calPhaseRef.current !== "recording") return;
      if (result.hands.length === 0) return;
      const session = calSessionRef.current;
      if (!session) return;
      const hand = pickPrimaryHand(result.hands);
      const norm = normalizeHand(hand.landmarks, hand.handedness);
      const letter = ALPHABET[calIdxRef.current];
      session.push(letter, norm);
      const count = session.recordedCount(letter);
      setCalRecordedCount(count);
      if (count >= CALIBRATION_FRAMES_PER_LETTER) {
        advanceCalibration();
      }
      return;
    }

    // TRANSLATION PATH
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
  }, [advanceCalibration]);

  const startCalibration = () => {
    calSessionRef.current = new CalibrationSession(ALPHABET);
    setCalIdx(0);
    setCalPhase("preview");
    setCalRecordedCount(0);
    setMode("calibrate");
    bufferRef.current?.clear();
    smootherRef.current?.reset();
  };

  const cancelCalibration = () => {
    calSessionRef.current = null;
    setMode("translate");
  };

  const finishCalibration = () => {
    const session = calSessionRef.current;
    if (!session) {
      setMode("translate");
      return;
    }
    const data = session.build();
    if (Object.keys(data.prototypes).length === 0) {
      setMode("translate");
      return;
    }
    CalibratedClassifier.saveToStorage(data);
    const cls = CalibratedClassifier.fromData(data);
    classifierRef.current = cls;
    setClassifierState({ kind: "calibrated", cls, sampleCounts: data.sampleCounts });
    calSessionRef.current = null;
    setMode("translate");
  };

  const clearCalibration = () => {
    CalibratedClassifier.clearStorage();
    setStarted(false); // forces classifier reload via the started effect
    setTimeout(() => setStarted(true), 0);
  };

  const skipLetter = () => {
    calSessionRef.current?.reset(ALPHABET[calIdxRef.current]);
    advanceCalibration();
  };

  const retryLetter = () => {
    calSessionRef.current?.reset(ALPHABET[calIdxRef.current]);
    setCalRecordedCount(0);
    setCalPhase("preview");
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

  const recordedLetters = useMemo(
    () => calSessionRef.current?.recordedLetters().length ?? 0,
    // Re-read whenever calIdx or phase changes — covers all the relevant moments.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [calIdx, calPhase],
  );

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
                index={calIdx}
                phase={calPhase}
                recordedCount={calRecordedCount}
                framesPerLetter={CALIBRATION_FRAMES_PER_LETTER}
                recordedLetters={recordedLetters}
                onSkip={skipLetter}
                onRetry={retryLetter}
                onCancel={cancelCalibration}
                onFinish={finishCalibration}
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
        onCalibrate={startCalibration}
        onClearCalibration={clearCalibration}
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
      <div className="text-xs text-cyan-300">Calibration in progress — stay still and follow the prompts.</div>
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
        <span className="text-emerald-300">Using your personal calibration ({letterCount} letters).</span>
        <button type="button" onClick={onCalibrate} className="text-cyan-300 hover:text-cyan-200 underline-offset-2 hover:underline">
          Recalibrate
        </button>
        <button type="button" onClick={onClearCalibration} className="text-zinc-500 hover:text-zinc-300 underline-offset-2 hover:underline">
          Use trained model instead
        </button>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
      <span>Using the trained model. Accuracy varies by hand & lighting.</span>
      <button type="button" onClick={onCalibrate} className="text-cyan-300 hover:text-cyan-200 underline-offset-2 hover:underline">
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
      <div className={pct >= 70 ? "h-full bg-emerald-500" : "h-full bg-amber-500"} style={{ width: `${pct}%` }} />
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
