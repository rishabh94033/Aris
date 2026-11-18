"""Speak with Friday.

The path is: microphone, Whisper, the same chat as the terminal, then Piper.
Run it with `python -m friday voice` from the Python 3.14 environment.
"""

import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

from friday.chat import handle_turn
from friday.persona import system_prompt
from friday.tts import gpt_ready, gpt_speech

ROOT = Path(__file__).resolve().parents[1]
VOICE_NAME = "en_GB-alan-medium"
VOICE_PATH = ROOT / "voices" / f"{VOICE_NAME}.onnx"
SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # Whisper expects 16 kHz audio.

STOP_WORDS = {"exit", "quit", "stop", "goodbye", "good bye"}
NAME = re.compile(r"\bfriday\b", re.IGNORECASE)
# 1.0 is Piper's normal pace. 1.3 speaks 30% faster.
SPEECH_RATE = 1.3
# Same speaker, about one semitone higher, so the voice is a little less deep.
PITCH = 1.08


def ensure_voice() -> None:
    """Download the Piper voice file once. It is the spoken voice, as the 3B file is the brain."""
    if VOICE_PATH.exists():
        return
    VOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the voice {VOICE_NAME} ...")
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", VOICE_NAME, "--download-dir", str(VOICE_PATH.parent)],
        check=True,
    )
    if not VOICE_PATH.exists():
        raise SystemExit(f"Piper did not write {VOICE_PATH}.")


def room_threshold() -> float:
    """Listen briefly on the headset microphone and set the volume that counts as speech."""
    print("Measuring the room. Stay quiet, Sir.")
    quiet = sd.rec(int(SAMPLE_RATE * 0.4), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    noise = float(np.sqrt(np.mean(quiet**2)))
    return max(noise * 4, 0.012)


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block**2)))


def record_utterance(threshold: float) -> np.ndarray:
    """Record until Sir speaks and then pauses. Returns silence if nobody speaks."""
    print("Listening, Sir. Speak, then pause.")
    preroll: deque[np.ndarray] = deque(maxlen=6)
    frames: list[np.ndarray] = []
    heard = False
    quiet_frames = 0
    waited = 0
    quiet_needed = 800 // FRAME_MS
    wait_limit = 8000 // FRAME_MS
    max_frames = 20_000 // FRAME_MS

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME) as stream:
        while True:
            block, _overflowed = stream.read(FRAME)
            level = _rms(block)
            if level >= threshold:
                if not heard:
                    frames.extend(preroll)
                    heard = True
                frames.append(block.copy())
                quiet_frames = 0
            elif heard:
                frames.append(block.copy())
                quiet_frames += 1
                if quiet_frames >= quiet_needed:
                    break
            else:
                preroll.append(block.copy())
                waited += 1
                if waited >= wait_limit:
                    break
            if len(frames) >= max_frames:
                break

    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames).reshape(-1)


def transcribe(ear: WhisperModel, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    segments, _info = ear.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def request_for_friday(text: str) -> str | None:
    """The words after his name, or None when the line does not address him."""
    if not NAME.search(text):
        return None
    request = NAME.sub(" ", text)
    request = re.sub(r"\s+", " ", request).strip(" ,.?!")
    return request


def for_speech(text: str) -> str:
    """Make one continuous line.

    Piper starts a new take at every full stop. Each take ends sharp, then leaves a gap,
    and the next take does not match the voice. Commas stay inside the same take.
    """
    spoken = text.replace("*", "").replace("#", "").strip()
    for mark in ".!?;:":
        spoken = spoken.replace(mark, ",")
    while ",," in spoken:
        spoken = spoken.replace(",,", ",")
    return spoken.strip(" ,")


def _finish(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Cut the dead tail after the last word and fade the ending."""
    window = int(sample_rate * 0.02)
    last_voiced = 0
    for start in range(0, len(audio) - window, window):
        piece = audio[start : start + window]
        if float(np.sqrt(np.mean(piece**2))) > 0.02:
            last_voiced = start + window
    end = min(len(audio), last_voiced + int(sample_rate * 0.06))
    audio = audio[:end]
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak * 0.9
    fade = min(len(audio), int(sample_rate * 0.04))
    audio = audio.copy()
    audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return audio


def _speaker_rate() -> int:
    """Wait until the headphones are back in music mode, then use that rate.

    The headset microphone puts them in phone-call mode. Playing during the change
    is what bends the first words. The pause happens before the sentence, in silence.
    """
    time.sleep(0.35)
    deadline = time.monotonic() + 1.0
    rate = 44_100
    while True:
        rate = int(sd.query_devices(kind="output")["default_samplerate"])
        if rate >= 32_000 or time.monotonic() >= deadline:
            return rate
        time.sleep(0.05)


def _fit_speaker(
    audio: np.ndarray, source_rate: int, speaker_rate: int, pitch: float = PITCH
) -> np.ndarray:
    """Move spoken audio onto the speaker rate without asking the headphones to change mode."""
    heard_rate = source_rate * pitch
    duration = len(audio) / heard_rate
    target = max(1, int(round(duration * speaker_rate)))
    if len(audio) == 1:
        return np.full(target, float(audio[0]), dtype=np.float32)
    return np.interp(
        np.linspace(0, len(audio) - 1, target),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _play(audio: np.ndarray, source_rate: int, pitch: float) -> None:
    rate = _speaker_rate()
    sd.play(_fit_speaker(audio, source_rate, rate, pitch), rate)
    sd.wait()


def speak(voice: PiperVoice | None, text: str) -> None:
    """Turn Friday's reply into sound and play it after the headphones have settled."""
    if voice is None:
        spoken = text.replace("*", "").replace("#", "").strip()
        if not spoken:
            return
        try:
            audio, rate = gpt_speech(spoken)
        except OSError as error:
            print(error)
            return
        _play(audio, rate, pitch=1.0)
        return

    spoken = for_speech(text)
    if not spoken:
        return
    # Playing higher raises the pitch and also speeds the audio up.
    # The length scale gives that speed back, so the heard pace stays SPEECH_RATE.
    chunks = list(
        voice.synthesize(
            spoken,
            syn_config=SynthesisConfig(
                length_scale=PITCH / SPEECH_RATE,
                noise_scale=0.0,
                noise_w_scale=0.0,
                normalize_audio=False,
            ),
        )
    )
    if not chunks:
        return
    audio = _finish(
        np.concatenate([chunk.audio_float_array for chunk in chunks]),
        chunks[0].sample_rate,
    )
    if audio.size == 0:
        return
    _play(audio, chunks[0].sample_rate, PITCH)


def main() -> None:
    try:
        _listen()
    except sd.PortAudioError as error:
        print("The microphone or speaker is not available.")
        print(error)
        print("On this Mac, allow the microphone for your terminal in System Settings, Privacy & Security.")
        raise SystemExit(1) from error


def _listen() -> None:
    use_gpt = gpt_ready()
    if use_gpt:
        voice = None
        print("Voice: ChatGPT cedar (gpt-4o-mini-tts).")
    else:
        ensure_voice()
        voice = PiperVoice.load(VOICE_PATH)
        print("Voice: Piper. Add OPENAI_API_KEY to .env to use the ChatGPT voice.")
    print("Loading the ear. The first run downloads the Whisper base model.")
    ear = WhisperModel("base", device="cpu", compute_type="int8")
    threshold = room_threshold()
    messages = [{"role": "system", "content": system_prompt()}]
    pending: str | None = None
    print("Say Friday, then the request. Ctrl+C stops immediately.")

    while True:
        try:
            audio = record_utterance(threshold)
            heard = transcribe(ear, audio)
        except KeyboardInterrupt:
            print()
            break
        if not heard:
            print("I heard nothing.")
            continue
        print(f"Sir: {heard}")
        if pending:
            named = request_for_friday(heard)
            answer = named if named is not None else heard
            reply, pending = handle_turn(messages, answer, pending)
            if reply:
                print(f"Friday: {reply}")
                speak(voice, reply)
            continue
        request = request_for_friday(heard)
        if request is None:
            print("No name in that line. Staying quiet.")
            continue
        if request.lower() in STOP_WORDS:
            speak(voice, "Goodbye, Sir.")
            break
        if not request:
            speak(voice, "Yes, Sir?")
            continue
        reply, pending = handle_turn(messages, request, None)
        if not reply:
            continue
        print(f"Friday: {reply}")
        speak(voice, reply)
