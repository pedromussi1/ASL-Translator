export interface SpeakOptions {
  voiceId?: string;
  rate?: number;   // 0.1 .. 10
  pitch?: number;  // 0 .. 2
  volume?: number; // 0 .. 1
  lang?: string;
}

export interface VoiceInfo {
  id: string;
  name: string;
  lang: string;
  default?: boolean;
}

export interface TTSProvider {
  /** Stable identifier for telemetry / settings persistence. */
  readonly id: string;

  /** Human-readable label for UI. */
  readonly name: string;

  /** Whether this provider can run in the current environment. */
  isAvailable(): boolean;

  /** List voices the provider exposes. May be async (e.g. cloud TTS). */
  listVoices(): Promise<VoiceInfo[]>;

  /**
   * Speak the text. Resolves when speech finishes (or is cancelled). Implementations
   * MUST cancel any in-flight utterance before starting a new one (barge-in).
   */
  speak(text: string, opts?: SpeakOptions): Promise<void>;

  /** Cancel the current and any queued utterances. Idempotent. */
  cancel(): void;
}
