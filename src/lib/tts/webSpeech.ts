import type { SpeakOptions, TTSProvider, VoiceInfo } from "./provider";

/**
 * Web Speech API provider. Available in all evergreen browsers.
 * Quality varies by OS — Edge/Windows neural voices are decent;
 * macOS Siri voices are good; Linux is rough.
 */
export class WebSpeechProvider implements TTSProvider {
  readonly id = "web-speech";
  readonly name = "Browser (Web Speech API)";

  private currentUtterance: SpeechSynthesisUtterance | null = null;

  isAvailable(): boolean {
    return typeof window !== "undefined" && "speechSynthesis" in window;
  }

  async listVoices(): Promise<VoiceInfo[]> {
    if (!this.isAvailable()) return [];
    const synth = window.speechSynthesis;

    let voices = synth.getVoices();
    if (voices.length === 0) {
      // Some browsers populate voices asynchronously.
      voices = await new Promise<SpeechSynthesisVoice[]>((resolve) => {
        const onChange = () => {
          synth.removeEventListener("voiceschanged", onChange);
          resolve(synth.getVoices());
        };
        synth.addEventListener("voiceschanged", onChange);
        // Belt-and-suspenders timeout.
        setTimeout(() => resolve(synth.getVoices()), 1000);
      });
    }

    return voices.map((v) => ({
      id: v.voiceURI,
      name: v.name,
      lang: v.lang,
      default: v.default,
    }));
  }

  async speak(text: string, opts: SpeakOptions = {}): Promise<void> {
    if (!this.isAvailable() || !text.trim()) return;
    const synth = window.speechSynthesis;

    // Barge-in: cancel anything already in flight.
    this.cancel();

    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = clamp(opts.rate ?? 1, 0.1, 10);
    utter.pitch = clamp(opts.pitch ?? 1, 0, 2);
    utter.volume = clamp(opts.volume ?? 1, 0, 1);
    utter.lang = opts.lang ?? "en-US";

    if (opts.voiceId) {
      const match = synth.getVoices().find((v) => v.voiceURI === opts.voiceId);
      if (match) utter.voice = match;
    }

    this.currentUtterance = utter;

    return new Promise<void>((resolve) => {
      const done = () => {
        if (this.currentUtterance === utter) this.currentUtterance = null;
        resolve();
      };
      utter.onend = done;
      utter.onerror = done;
      synth.speak(utter);
    });
  }

  cancel(): void {
    if (!this.isAvailable()) return;
    window.speechSynthesis.cancel();
    this.currentUtterance = null;
  }
}

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}
