"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CameraView, type CameraStatus } from "@/components/CameraView";
import { AlphabetClassifier } from "@/lib/recognition/classifier";
import {
  DynamicSignDetector,
  type MatcherScores,
} from "@/lib/recognition/dynamicDetector";
import { MotionMonitor } from "@/lib/recognition/motionMonitor";
import { normalizeHand } from "@/lib/recognition/normalize";
import { PredictionSmoother } from "@/lib/recognition/smoother";
import type { DetectedHand, FrameResult, Prediction } from "@/lib/recognition/types";
import { WordBuffer } from "@/lib/recognition/wordBuffer";
import { WebSpeechProvider } from "@/lib/tts/webSpeech";

type ClassifierState =
  | { kind: "loading" }
  | { kind: "ready" }
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

  const classifierRef = useRef<AlphabetClassifier | null>(null);
  const smootherRef = useRef<PredictionSmoother | null>(null);
  const bufferRef = useRef<WordBuffer | null>(null);
  const dynamicRef = useRef<DynamicSignDetector | null>(null);
  const ttsRef = useRef<WebSpeechProvider | null>(null);
  const inflightRef = useRef(false);
  const [lastDynamic, setLastDynamic] = useState<string | null>(null);
  const [scores, setScores] = useState<MatcherScores>({ J: 0, Z: 0 });
  const [inMotion, setInMotion] = useState(false);
  const scoresTickRef = useRef(0);
  const motionRef = useRef<MotionMonitor | null>(null);

  // Initialize the singletons once.
  useEffect(() => {
    bufferRef.current = new WordBuffer();
    smootherRef.current = new PredictionSmoother(5);
    dynamicRef.current = new DynamicSignDetector();
    motionRef.current = new MotionMonitor();
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

  // Lazy-load the classifier when the user starts the camera.
  useEffect(() => {
    if (!started) return;
    let cancelled = false;
    setClassifierState({ kind: "loading" });
    AlphabetClassifier.load({
      modelUrl: "/models/asl_alphabet_v1/alphabet.onnx",
      labelsUrl: "/models/asl_alphabet_v1/labels.json",
    })
      .then((c) => {
        if (cancelled) {
          c.close();
          return;
        }
        classifierRef.current = c;
        setClassifierState({ kind: "ready" });
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
      classifierRef.current?.close();
      classifierRef.current = null;
    };
  }, [started]);

  const onFrame = useCallback(async (result: FrameResult) => {
    const buf = bufferRef.current;
    const smoother = smootherRef.current;
    const dynamic = dynamicRef.current;
    const cls = classifierRef.current;
    if (!buf || !smoother || !dynamic) return;

    const hand = result.hands.length > 0 ? pickPrimaryHand(result.hands) : null;

    // Dynamic-sign detector runs first. If it fires, we bypass the letter
    // pipeline — committing whatever's in the word buffer, then appending the
    // dynamic letter and speaking it.
    const dyn = dynamic.push(hand, result.timestampMs);
    if (dyn) {
      const pending = buf.currentWord;
      if (pending) {
        setTranscript((prev) => [...prev, pending, dyn.label]);
      } else {
        setTranscript((prev) => [...prev, dyn.label]);
      }
      buf.clear();
      smoother.reset();
      setCurrentWord("");
      setTentative({ letter: null, conf: 0 });
      setLastDynamic(dyn.label);
      ttsRef.current?.cancel();
      ttsRef.current?.speak(dyn.label);
      return;
    }

    // Suppress static-letter commits while the user's hand is moving so a
    // dynamic sign (J starts as pinky-out / "I", Z as index-out / "D")
    // can't squeak through as a letter before the dynamic detector finishes
    // its 1.5s buffer.
    const userInMotion = motionRef.current?.push(hand) ?? false;

    // Throttle the score-panel state updates to ~6 Hz so we don't re-render
    // every frame.
    scoresTickRef.current = (scoresTickRef.current + 1) % 5;
    if (scoresTickRef.current === 0) {
      setScores(dynamic.getLastScores());
      setInMotion(userInMotion);
    }

    if (!cls || !hand || userInMotion) {
      buf.feed(smoother.push(null), result.timestampMs);
      return;
    }
    if (inflightRef.current) return;
    inflightRef.current = true;
    try {
      const norm = normalizeHand(hand.landmarks, hand.handedness);
      const raw: Prediction = await cls.recognize(norm);
      buf.feed(smoother.push(raw), result.timestampMs);
    } finally {
      inflightRef.current = false;
    }
  }, []);

  const handleClear = () => {
    bufferRef.current?.clear();
    smootherRef.current?.reset();
    dynamicRef.current?.reset();
    motionRef.current?.reset();
    setTranscript([]);
    setLastDynamic(null);
    setInMotion(false);
    ttsRef.current?.cancel();
  };

  const handleSpace = () => {
    // Force-end the current word.
    bufferRef.current?.feed(null, performance.now() + 10_000);
  };

  return (
    <main className="flex flex-1 w-full mx-auto max-w-5xl flex-col gap-6 p-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">ASL Translator</h1>
        <span className="text-xs uppercase tracking-widest text-zinc-400">
          v0.1 · alphabet
        </span>
      </header>

      <p className="text-sm text-zinc-400 max-w-2xl">
        Phase 1: fingerspelling alphabet A–Z, plus the two motion letters{" "}
        <span className="font-mono">J</span> and <span className="font-mono">Z</span>.{" "}
        <span className="text-zinc-300">
          Hold each letter steady for ~300ms; words commit when you drop your hand.
        </span>{" "}
        Everything runs in your browser — no frames leave your device.
      </p>

      <section className="rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800 shadow-lg">
        {started ? (
          <CameraView onStatusChange={setStatus} onFrame={onFrame} className="aspect-video" />
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

      <ModelBanner state={classifierState} started={started} />

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
            {transcript.length > 0 ? transcript.join(" ") : (
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

      <DynamicScoresHUD scores={scores} inMotion={inMotion} started={started} />

      <StatusBar status={status} />

      <footer className="text-xs text-zinc-500 pt-4 border-t border-zinc-800">
        Personal learning project. Built on MediaPipe Tasks + ONNX Runtime Web.
      </footer>
    </main>
  );
}

function pickPrimaryHand(hands: DetectedHand[]): DetectedHand {
  // Pick the hand with the largest bounding-box area on screen.
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

function ModelBanner({ state, started }: { state: ClassifierState; started: boolean }) {
  if (!started) return null;
  if (state.kind === "loading") {
    return <div className="text-xs text-amber-300">Loading classifier…</div>;
  }
  if (state.kind === "error") {
    return <div className="text-xs text-rose-400">Classifier failed: {state.message}</div>;
  }
  return (
    <div className="text-xs text-zinc-500">
      Using stub classifier (random weights). Run{" "}
      <code className="font-mono text-zinc-300">python training/train_alphabet.py</code>{" "}
      for the real model.
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

function DynamicScoresHUD({
  scores,
  inMotion,
  started,
}: {
  scores: MatcherScores;
  inMotion: boolean;
  started: boolean;
}) {
  if (!started) return null;
  const labels: Array<keyof MatcherScores> = ["J", "Z"];
  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-3 flex flex-col gap-2 text-xs">
      <div className="flex items-baseline justify-between gap-3">
        <span className="uppercase tracking-widest text-zinc-500">Motion-letter scores</span>
        <span className={inMotion ? "text-cyan-300" : "text-zinc-500"}>
          motion: {inMotion ? "yes (letters paused)" : "no"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {labels.map((l) => {
          const v = scores[l];
          const pct = Math.round(Math.max(0, Math.min(1, v)) * 100);
          const ramp =
            pct >= 70
              ? "bg-emerald-500"
              : pct >= 40
                ? "bg-amber-500"
                : "bg-zinc-700";
          return (
            <div key={l} className="flex flex-col gap-1 items-stretch">
              <span className="text-[10px] uppercase tracking-widest text-zinc-400">{l}</span>
              <div className="h-1.5 rounded bg-zinc-800 overflow-hidden">
                <div className={`h-full ${ramp}`} style={{ width: `${pct}%` }} />
              </div>
              <span className="text-[10px] font-mono text-zinc-300">{v.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
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
        <span
          className={status.handsDetected > 0 ? "text-emerald-400" : "text-zinc-500"}
        >
          {status.handsDetected}
        </span>
      </span>
    </div>
  );
}
