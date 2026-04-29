"use client";

import { useState } from "react";
import { CameraView, type CameraStatus } from "@/components/CameraView";

export default function Home() {
  const [started, setStarted] = useState(false);
  const [status, setStatus] = useState<CameraStatus>({ kind: "idle" });

  return (
    <main className="flex flex-1 w-full mx-auto max-w-5xl flex-col gap-6 p-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          ASL Translator
        </h1>
        <span className="text-xs uppercase tracking-widest text-zinc-400">
          v0.1 · alphabet
        </span>
      </header>

      <p className="text-sm text-zinc-400 max-w-2xl">
        Phase 1: fingerspelling alphabet. Hold each letter steady for ~300ms;
        words commit when you drop your hand. Everything runs in your browser —
        no frames leave your device.
      </p>

      <section className="rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800 shadow-lg">
        {started ? (
          <CameraView
            onStatusChange={setStatus}
            onFrame={() => {
              /* Day 8: feed to classifier */
            }}
            className="aspect-video"
          />
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

      <StatusBar status={status} />

      <footer className="text-xs text-zinc-500 pt-4 border-t border-zinc-800">
        Personal learning project. Built on MediaPipe Tasks + ONNX Runtime Web.
      </footer>
    </main>
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
    return (
      <div className="text-xs text-rose-400">Error: {status.error.message}</div>
    );
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
          className={
            status.handsDetected > 0 ? "text-emerald-400" : "text-zinc-500"
          }
        >
          {status.handsDetected}
        </span>
      </span>
    </div>
  );
}
