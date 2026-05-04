# ASL Translator

A real-time American Sign Language → text + speech translator that runs **entirely in your browser**. No frames ever leave your device.

> **Status:** Phase 1 in progress. Personal learning project.

## What works today

- **Fingerspelling A–Z** via a tiny MLP trained on Kaggle ASL Alphabet (98.06% test accuracy). The browser runs the ONNX in WASM.
- **The two motion letters** via a heuristic detector that combines hand-shape + landmark trajectory:
  - **J** — pinky out, traces a J
  - **Z** — index out, draws a Z in the air
- **Motion suppression** (`MotionMonitor`) that pauses static letter commits while the hand is moving so an `I` doesn't sneak through ahead of a `J`.
- **5-frame temporal smoother** between the classifier and the word buffer to drop single-frame flicker.
- **Word buffer state machine** that handles repeated letters via a 200ms release threshold (HELLO, BOOK work).
- **Web Speech API TTS** behind a `TTSProvider` interface so a cloud TTS can drop in later.
- **Live HUD** showing motion state and J / Z matcher scores.

## Phased roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Fingerspelling A–Z including the motion letters J and Z, all on-device | shipping incrementally |
| 2 | Vocabulary recognition (~250–1000 signs) + Tauri 2 desktop wrapper | planned |
| 3 | Continuous conversational ASL + iOS (Swift + CoreML) | research / future |

## Stack

- **Web:** Next.js 16 (App Router) + TypeScript + Tailwind CSS
- **Hand tracking:** [`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) `HandLandmarker` — GPU delegate, CPU fallback
- **Static-letter inference:** [`onnxruntime-web`](https://onnxruntime.ai/docs/get-started/with-javascript.html) (WASM EP; WebGPU when available)
- **Motion letters (J, Z):** purpose-built heuristic detector — hand-shape + landmark trajectory. No LSTM; see [ADR-006](docs/DECISIONS.md) for why.
- **Speech output:** Web Speech API behind a `TTSProvider` interface
- **Tests:** Vitest (36 unit tests as of latest)
- **Training (separate, Python):** PyTorch + MediaPipe Tasks API for landmark extraction, exported to ONNX

## Run locally

```sh
npm install
npm run dev
# open http://localhost:3000 and grant camera access
```

## Repo layout (Phase 1, single Next.js app — no monorepo until Phase 2)

```
src/
├── app/
│   └── page.tsx                  # main page: camera + HUD + transcript
├── components/
│   ├── CameraView.tsx            # owns the rAF loop, HandLandmarker + FaceDetector engines
│   └── LandmarkOverlay.tsx       # canvas: hand wireframe + face keypoints
└── lib/
    ├── camera/index.ts           # getUserMedia wrapper, brightness check
    ├── recognition/
    │   ├── handLandmarker.ts     # MediaPipe Tasks HandLandmarker wrapper
    │   ├── normalize.ts          # wrist-relative, scale-invariant features (mirror left hand)
    │   ├── classifier.ts         # ONNX Runtime Web wrapper for the static MLP
    │   ├── smoother.ts           # 5-frame sliding-window prediction smoother
    │   ├── motionMonitor.ts      # pauses static letters while the hand is in motion
    │   ├── wordBuffer.ts         # state machine: letters → words → speech triggers
    │   ├── dynamicDetector.ts    # heuristic detector for J and Z
    │   └── types.ts              # shared types
    └── tts/
        ├── provider.ts           # TTSProvider interface
        └── webSpeech.ts          # Web Speech API implementation with barge-in

public/models/asl_alphabet_v1/   # alphabet.onnx (71.5 KB) + labels.json (26 classes)
training/                         # Python pipeline — see training/README.md
docs/DECISIONS.md                 # ADR log (architecture decisions + rationale)
```

## Scripts

```sh
npm run dev          # Next.js dev server (Turbopack)
npm run build        # production build
npm run lint         # ESLint
npm test             # Vitest, single run
npm run test:watch   # Vitest, watch mode
```

## Training pipeline

The static-letter classifier is trained from scratch in Python; see [`training/README.md`](training/README.md) for the full walkthrough. Summary:

1. `synthetic_warmup.py` — produces a stub ONNX with random weights so the browser pipeline can be wired up before real training is done.
2. `extract_landmarks.py` — runs MediaPipe Tasks HandLandmarker over the Kaggle ASL Alphabet dataset, normalizes (matching the browser's `normalize.ts`), and writes a single `landmarks.npz`.
3. `train_alphabet.py` — trains a 2-layer MLP with online data augmentation (random 3D rotation ±15°, scale ±10%, per-landmark Gaussian noise) and exports a single self-contained ONNX with `dynamo=False`.

The browser's `AlphabetClassifier` and the Python `train_alphabet.py` agree on the IO contract: `[1, 63] float32 input → [1, 26] float32 logits` with sibling `labels.json`.

## Architecture notes

The big design decisions and their rationale live in [`docs/DECISIONS.md`](docs/DECISIONS.md). Key calls:

- **Single Next.js app for Phase 1** — monorepo overhead deferred until Phase 2 ([ADR-001](docs/DECISIONS.md)).
- **All inference on-device.** Privacy is a real product feature here, not a footnote.
- **Heuristic over LSTM for the motion letters J and Z** — handcrafted matchers were faster to ship and the user explicitly rejected a self-recording UX ([ADR-006](docs/DECISIONS.md)).
- **Face landmarks were tried and removed** — added FaceDetector to anchor HELLO/THANK YOU/YES, but those signs were unreliable in practice and have been dropped from Phase 1. Phase 1 is now fingerspelling-focused ([ADR-007](docs/DECISIONS.md)).
- **Online data augmentation** in static-letter training — gave up <0.3% in-distribution accuracy for cross-signer robustness ([ADR-008](docs/DECISIONS.md)).
- **Personal calibration was tried and reverted** — kept in git history; the better lever turned out to be a more diverse dataset rather than per-user fine-tuning ([ADR-009](docs/DECISIONS.md)).
- **Motion suppression** prevents an `I` from committing as a letter while the user is mid-J ([ADR-012](docs/DECISIONS.md)).
- **Pluggable TTS interface** so Phase 2 can drop in cloud TTS without refactors.

## Acknowledgements

- Hand + face landmark detection: [Google MediaPipe](https://developers.google.com/mediapipe).
- Static-classifier training data: [Kaggle ASL Alphabet (`grassknoted/asl-alphabet`)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — GPL-2.0 licensed; usable for personal/research, not for commercial redistribution.

## License

[MIT](LICENSE)
