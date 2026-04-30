# Architecture Decision Records

Lightweight ADRs. One entry per decision, in chronological order.

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

---

## ADR-006 — Heuristic dynamic-sign detector, not an LSTM

**Date:** 2026-04-30
**Status:** Accepted (revisit if vocabulary expands beyond ~10 signs)

**Context.** Phase 1 needs to recognize five dynamic signs (J, Z, YES, HELLO, THANK YOU). Three implementation paths were on the table:

1. Self-recorded LSTM — quick recording UX (~10 min user effort), then PyTorch training and ONNX export.
2. Public-dataset LSTM — WLASL has HELLO/THANK YOU/YES, but J and Z are letters and live in different fingerspelling datasets. Multi-dataset wrangling for five signs is poor ROI.
3. Handcrafted heuristic — landmark-trajectory matchers per sign, no training data needed.

**Decision.** Build a heuristic [`DynamicSignDetector`](../src/lib/recognition/dynamicDetector.ts). Each sign has a hand-shape predicate (e.g. pinky-out for J) plus a motion-pattern predicate (e.g. pinky tip drops then curves toward thumb side). A 1.5s rolling buffer feeds all matchers per push; a 1.5s cooldown prevents repeat firings.

**Why.** Five specific signs is small enough to handcraft well, the user explicitly rejected a self-recording UX after testing personal calibration (see [ADR-009](#adr-009)), and a heuristic ships in an afternoon vs days of dataset prep. The interface (`push(hand, face, t) → DynamicResult | null`) doesn't lock us into heuristics — an LSTM-backed implementation can drop in behind the same shape later.

**Trade-off.** Doesn't scale beyond ~10 signs without becoming unmaintainable. HELLO and THANK YOU heuristics are intrinsically rough without face/pose context (mitigated by [ADR-007](#adr-007)).

---

## ADR-007 — `FaceDetector` (6 keypoints) over `FaceLandmarker` (478 points)

**Date:** 2026-04-30
**Status:** Accepted

**Context.** The HELLO and THANK YOU dynamic-sign matchers need to anchor "near the temple" and "near the mouth" to the user's actual head, not absolute screen y-coordinate. MediaPipe Tasks-Vision Web exposes two face options:

- `FaceDetector` — BlazeFace short-range, returns bounding box + 6 keypoints (eyes, nose tip, mouth center, ear tragions). ~3 MB model.
- `FaceLandmarker` — 478 face mesh points. Order of magnitude larger model, slower per-frame.

**Decision.** Use `FaceDetector`. Run it every 3rd frame (face moves slowly) in [`CameraView`](../src/components/CameraView.tsx) alongside `HandLandmarker`, share the same WASM `FilesetResolver`.

**Why.** All we need is "where is the head, where is the mouth, where is the same-side ear" — those are exactly four of the six BlazeFace keypoints. The 478-point mesh would be overkill for the question we're asking and would visibly burn battery for nothing.

**Trade-off.** No precise mouth-shape or brow-position info, which would matter in Phase 3 when ASL non-manual markers (eyebrow raise = question) drive grammar. Phase 3 will likely upgrade to `FaceLandmarker` or pose-only approaches.

**Note.** The JS SDK does not populate `keypoint.label` strings — the wrapper in [`faceDetector.ts`](../src/lib/recognition/faceDetector.ts) indexes by position (BlazeFace's documented order: 0=right eye, 1=left eye, 2=nose tip, 3=mouth center, 4=right ear, 5=left ear).

---

## ADR-008 — Online data augmentation in static-classifier training

**Date:** 2026-04-30
**Status:** Accepted

**Context.** First training run hit 98.34% on a held-out split of the Kaggle ASL Alphabet dataset, but accuracy on a different signer's hand (different proportions, lighting, camera angle) was much rougher — common confusions in M/N/S, T/A, F/B. The dataset has only one signer, so more samples didn't help cross-signer generalization.

**Decision.** Add online landmark augmentation in [`train_alphabet.py`](../training/train_alphabet.py): per batch, apply random 3D rotation about the wrist (±15°), uniform scale jitter (±10%), and per-landmark Gaussian noise (σ=0.01).

**Why.** Augmentation forces the MLP to learn shape features that don't depend on a specific hand size, camera angle, or normalization-precision artifact. Pays for itself in cross-signer accuracy at a small cost in same-signer accuracy.

**Trade-off.** Held-out accuracy dropped 98.34% → 98.06%. Acceptable: that's the regression we wanted (less overfit to the single training signer). Real-world cross-signer accuracy is harder to measure but qualitatively better.

---

## ADR-009 — Reverted: personal calibration as a per-user classifier

**Date:** 2026-04-30
**Status:** Reverted (commits `348a7e1`, `b1326ea` retained in history)

**Context.** Cross-signer accuracy gaps suggested per-user calibration could help: record the user signing each letter, build a nearest-prototype classifier from those samples. Two iterations were tried — sequential A→Z walkthrough, then a manual per-letter grid with save-as-you-go.

**Decision.** Reverted both calibration commits. The trained model + augmentation + smoother stack now has no per-user training step.

**Why the revert.** The recording UX was tedious in either form, and the resulting per-user classifier didn't deliver enough accuracy to justify the workflow cost. The user's words: *"I do not like the results."*

**What's preserved.** The `CalibratedClassifier` and `CalibrationOverlay` code lives in git at the reverted commits — recoverable if a future approach (multi-prototype per letter, on-device fine-tune of the ONNX, hybrid trained-plus-calibrated) is wanted.

**What we learned.** For a personal project the fastest gains come from improving the dataset (or augmentation/regularization on it), not from forcing the user to be a one-person training set.

---

## ADR-010 — 5-frame `PredictionSmoother` between classifier and word buffer

**Date:** 2026-04-30
**Status:** Accepted

**Context.** Per-frame static-classifier predictions are noisy: a stable hand pose can produce single-frame mispredictions (e.g. one frame of "T" while you're holding "A"). If those reach the word buffer, they can briefly contend for the stability lock and let through wrong letters under high-noise conditions.

**Decision.** Insert a 5-frame sliding-window [`PredictionSmoother`](../src/lib/recognition/smoother.ts) between the classifier and the word buffer. The smoothed output is the modal label across the window, with confidence averaged over agreeing frames; ties break toward higher confidence.

**Why.** Cheap (5-deep deque, O(window) per push), pure logic, fully tested in isolation. Drops single-frame flicker without changing the buffer state machine.

**Trade-off.** Adds ~5 frames of latency (~165 ms at 30 fps) before a letter prediction reaches the buffer. Imperceptible relative to the 300 ms `stableHoldMs` already required for letter commit, so net latency cost is effectively zero.

---

## ADR-011 — Heuristic shape thresholds tuned for real human hands

**Date:** 2026-04-30
**Status:** Accepted

**Context.** Initial dynamic-sign matchers used `EXT_THRESH = 0.92` and `CURL_THRESH = 0.83` for the finger-extension ratio (straight-line distance from MCP to tip ÷ kinked path through PIP/DIP/tip). These thresholds work for synthetic test fingers but real human hands rarely produce a "perfectly straight" finger — the open-hand check was silently failing for most users, blocking HELLO and THANK YOU before they could even score.

**Decision.** Lowered to `EXT_THRESH = 0.85`, `CURL_THRESH = 0.75`. Comfortable gap preserved between the two states; ambiguous middle band (0.75–0.85) gracefully scores as 0 in both predicates.

**Why.** Calibrated against actual MediaPipe output on a live hand. The synthetic test helper builds finger landmarks colinearly so its ratios are exactly 1.0 (extended) or 0.33 (curled) — those tests still pass with the lower thresholds. Real-world ratios sit in the 0.85–0.99 range for "extended" and 0.40–0.70 for "curled."

**Trade-off.** Slight risk of false positives on poses where a finger is in the ambiguous 0.75–0.85 band (e.g. a half-curled index in some fingerspelling letters). The shape-score ramp goes to 0 outside the bands, so impact is limited to motion-trajectory-positive false fires, which the buffer's other constraints (motion magnitude, direction, cooldown) further suppress.
