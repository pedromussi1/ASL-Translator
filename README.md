# ASL Translator

A real-time American Sign Language → text + speech translator that runs **entirely in your browser**. No frames ever leave your device.

> **Status:** Phase 1 in progress (alphabet / fingerspelling). Personal learning project.

## Phased roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Fingerspelling alphabet A–Z + 0–9 + a few common dynamic signs (J, Z, hello, thank you, yes) | in progress |
| 2 | Vocabulary recognition (~250–1000 signs) + Tauri 2 desktop wrapper | planned |
| 3 | Continuous conversational ASL + iOS (Swift + CoreML) | research / future |

## Stack

- **Web:** Next.js 16 (App Router) + TypeScript + Tailwind CSS
- **Hand tracking:** [`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) (HandLandmarker) — GPU delegate with WASM fallback
- **Inference:** [`onnxruntime-web`](https://onnxruntime.ai/docs/get-started/with-javascript.html) (WASM EP; WebGPU when available)
- **Speech output:** Web Speech API behind a `TTSProvider` interface (so we can plug ElevenLabs / Azure later)
- **Tests:** Vitest
- **Training (separate, Python):** PyTorch + MediaPipe Python for landmark extraction, exported to ONNX

## Run locally

```sh
npm install
npm run dev
# open http://localhost:3000 and grant camera access
```

## Repo layout (Phase 1, single app — no monorepo until Phase 2)

```
src/
├── app/                       # Next.js App Router routes
├── components/
│   ├── CameraView.tsx         # owns the rAF loop, brightness sampling, status
│   └── LandmarkOverlay.tsx    # canvas wireframe + joints, color-coded by handedness
└── lib/
    ├── camera/index.ts        # getUserMedia wrapper, brightness check, device list
    ├── recognition/
    │   ├── handLandmarker.ts  # MediaPipe Tasks-Vision wrapper (GPU→CPU fallback)
    │   ├── normalize.ts       # wrist-relative, scale-invariant features (mirror left hand)
    │   ├── types.ts           # shared types
    │   ├── classifier.ts      # ONNX Runtime Web wrapper                  [planned]
    │   ├── dynamicLstm.ts     # 30-frame LSTM for J/Z/hello/...           [planned]
    │   └── wordBuffer.ts      # state machine: letters → words → speech   [planned]
    └── tts/
        ├── provider.ts        # TTSProvider interface
        └── webSpeech.ts       # Web Speech API implementation with barge-in

public/models/                 # versioned ONNX artifacts (added during training)
training/                      # Python PyTorch training scripts
docs/DECISIONS.md              # ADR log (architecture decisions + rationale)
```

## Scripts

```sh
npm run dev          # Next.js dev server (Turbopack)
npm run build        # production build
npm run lint         # ESLint
npm test             # Vitest, single run
npm run test:watch   # Vitest, watch mode
```

## Architecture notes

The big design decisions and their rationale live in [`docs/DECISIONS.md`](docs/DECISIONS.md). Highlights:

- **Single Next.js app for Phase 1.** Monorepo overhead is deferred until Phase 2 actually needs `packages/core` for desktop reuse.
- **All inference on-device.** Privacy is a real product feature here, not a footnote.
- **Pluggable TTS interface from day one** so Phase 2 can drop in cloud TTS without refactors.
- **Public datasets only** (Kaggle ASL Alphabet for Phase 1) — and the project stays a personal learning effort. Commercializing would require a first-party dataset built with paid Deaf signers; that's deliberately out of scope.

## Acknowledgements

- Hand landmark detection: [Google MediaPipe](https://developers.google.com/mediapipe).
- Phase 1 training data (when added): [Kaggle ASL Alphabet datasets](https://www.kaggle.com/datasets) (CC-licensed variants).

## License

[MIT](LICENSE)
