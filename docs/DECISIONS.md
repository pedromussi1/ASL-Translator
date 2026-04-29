# Architecture Decision Records

Lightweight ADRs. One entry per decision. Latest first.

---

## ADR-001 — Single Next.js app, not a monorepo (Phase 1)

**Date:** 2026-04-29
**Status:** Accepted

**Context.** The project will eventually ship to web, Windows desktop, and iOS. A monorepo with `packages/core` would set up code sharing for the desktop wrapper from day one.

**Decision.** Phase 1 ships as a single Next.js app at the repo root. We extract a `packages/core` workspace at Phase 2, when the desktop (Tauri) wrapper actually needs to consume shared code.

**Why.** Monorepo overhead (workspace tooling, build orchestration, package boundaries) is real and pays off only when there's a second consumer. Phase 1 has one app. Premature workspace abstraction would slow the 2-week MVP without unlocking anything.

**Trade-off.** A future refactor will move `src/lib/recognition` and `src/lib/tts` into a shared package. That's a mechanical move, not a redesign — the module boundaries we're committing to now (see ADR-002) are workspace-ready.

---

## ADR-002 — Module boundaries: `recognition`, `tts`, `camera` are independent

**Date:** 2026-04-29
**Status:** Accepted

**Context.** Even without a workspace, we need module boundaries that survive the Phase 2 monorepo extraction.

**Decision.** Three independent libs under `src/lib/`:
- `camera/` — `getUserMedia` wrapper, readiness checks. Knows nothing about ML.
- `recognition/` — landmark extraction (MediaPipe), classifier (ONNX), word buffer. Consumes raw video frames; emits `{label, confidence}` and committed words. Knows nothing about audio output.
- `tts/` — `TTSProvider` interface + `WebSpeechProvider`. Consumes text; emits speech.

Components in `src/components/` wire these together. No cross-imports between the three libs.

**Why.** Each lib will move into its own package eventually. Keeping them decoupled now avoids tangled refactors later and makes each unit-testable in isolation.

---

## ADR-003 — `@mediapipe/tasks-vision` for hand tracking (not legacy MediaPipe)

**Date:** 2026-04-29
**Status:** Accepted

**Context.** MediaPipe ships two web bundles: legacy `@mediapipe/hands` (mature but deprecated) and `@mediapipe/tasks-vision` (the supported successor, exposes HandLandmarker via the Tasks API).

**Decision.** Use `@mediapipe/tasks-vision` `HandLandmarker` for Phase 1. Initialize with the GPU delegate; fall back to CPU automatically.

**Why.** Legacy bundle is on the deprecation path. Tasks API is what Google is actively maintaining and what new model variants ship through.

**Trade-off.** Tasks Web does NOT expose HolisticLandmarker — Phase 2 will need to compose Hand + Pose + Face Tasks separately, or export Holistic TFLite to ONNX. Deferred to Phase 2.

---

## ADR-004 — `onnxruntime-web` for the trained classifier (not TF.js)

**Date:** 2026-04-29
**Status:** Accepted

**Context.** We need to run a small MLP classifier (~50KB) in the browser on landmark inputs.

**Decision.** Train in PyTorch, export to ONNX, run with `onnxruntime-web`. Use the WASM execution provider by default; opt into WebGPU when feature-detected.

**Why.** PyTorch is the dominant framework for ML research and has cleaner training ergonomics than TF.js / Keras. ONNX is the cross-runtime standard — same artifact runs in browser (`onnxruntime-web`), Tauri/desktop (`onnxruntime-node`), and iOS (CoreML conversion via `coremltools`). One model, three platforms.

**Trade-off.** Slightly larger runtime than TF.js. Mitigated by lazy-loading and the fact that classifier itself is tiny.

---

## ADR-005 — npm, not pnpm (for Phase 1 only)

**Date:** 2026-04-29
**Status:** Accepted (revisit at Phase 2)

**Context.** The plan referenced pnpm; the dev environment had only npm installed.

**Decision.** Use npm for Phase 1. Switch to pnpm when we extract a workspace at Phase 2 (where pnpm's workspace support shines).

**Why.** No workspaces in Phase 1 means npm is sufficient. Avoids a global install for a single-app project.
