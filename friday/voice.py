"""Speak with Friday.

The path is: microphone, Distil-Whisper, the same chat as the terminal, then a voice.
The ear is distil-large-v3.5. It runs on this Mac and costs nothing.
With OPENAI_API_KEY the speaking voice is gpt-4o-mini-tts. Without the key it is Piper.
Run it with `python -m friday voice` from the Python 3.14 environment.
"""

import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

from friday.chat import handle_turn, warm
from friday.persona import system_prompt
from friday.tts import gpt_ready, gpt_speech

ROOT = Path(__file__).resolve().parents[1]
VOICE_NAME = "en_GB-jenny_dioco-medium"
VOICE_PATH = ROOT / "voices" / f"{VOICE_NAME}.onnx"
# Best free English ear that still fits beside the 3B brain on 8 GB.
# Full Whisper large is more accurate on long recordings and about twice this size.
EAR_MODEL = "distil-large-v3.5"
SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # Whisper expects 16 kHz audio.

STOP_WORDS = {"exit", "quit", "stop", "goodbye", "good bye"}
# After she answers, a line that starts within this many seconds is for her, even without the name.
FOLLOW_SECONDS = 10
NAME = re.compile(r"\bfriday\b", re.IGNORECASE)
# 1.0 is Piper's normal pace. 1.1 speaks 10% faster, the pace of the sample Sir chose.
SPEECH_RATE = 1.1
# 1.0 keeps her natural pitch. Above 1.0 plays her higher.
PITCH = 1.0


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


def use_mac_microphone() -> bool:
    """Listen with the Mac's built-in microphone, whatever the speaker is.

    A Bluetooth headset switches to phone-call mode whenever its own microphone is on,
    and words spoken during the switch are lost. With the Mac's microphone the headset
    only plays, so it never switches. True means the Mac microphone was found.
    """
    if sys.platform != "darwin":
        return False
    for index, device in enumerate(sd.query_devices()):
        name = device["name"].lower()
        if device["max_input_channels"] > 0 and ("macbook" in name or "built-in" in name):
            sd.default.device = (index, sd.default.device[1])
            print(f"Ear: {device['name']}.")
            return True
    print("The Mac microphone was not found. Using the default microphone.")
    return False


def room_threshold() -> float:
    """Listen briefly on the microphone and set the volume that counts as speech."""
    print("Measuring the room. Stay quiet, Sir.")
    quiet = sd.rec(int(SAMPLE_RATE * 0.4), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    noise = float(np.sqrt(np.mean(quiet**2)))
    return max(noise * 4, 0.012)


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block**2)))


# People pause between words for up to about half a second. 0.8s of quiet ends the line.
QUIET_TO_END_MS = 800
# Keep this much sound from before the level rose, so a soft first word is not cut.
PREROLL_MS = 300
# Shorter than this is a click or a breath, not a word.
MIN_SPEECH_MS = 250


class Endpointer:
    """Decide, one 30 ms block at a time, where a spoken line starts and ends.

    Speech starts when a block is louder than the room threshold.
    Once he is speaking, a softer level still counts as speech, so quiet word
    endings and short pauses do not end the line.
    """

    def __init__(self, threshold: float) -> None:
        self.start_level = threshold
        self.keep_level = threshold * 0.5
        self.preroll: deque[np.ndarray] = deque(maxlen=PREROLL_MS // FRAME_MS)
        self.frames: list[np.ndarray] = []
        self.speaking = False
        self.voiced = 0
        self.quiet = 0
        self.waited = 0

    def feed(self, block: np.ndarray) -> bool:
        """Take one block. True means the line is over, or nobody spoke in time."""
        level = _rms(block)
        if not self.speaking:
            if level >= self.start_level:
                self.speaking = True
                self.frames.extend(self.preroll)
                self.frames.append(block)
                self.voiced = 1
                return False
            self.preroll.append(block)
            self.waited += 1
            return self.waited >= 8000 // FRAME_MS
        self.frames.append(block)
        if level >= self.keep_level:
            self.voiced += 1
            self.quiet = 0
        else:
            self.quiet += 1
        return self.quiet >= QUIET_TO_END_MS // FRAME_MS or len(self.frames) >= 20_000 // FRAME_MS

    def audio(self) -> np.ndarray:
        if self.voiced * FRAME_MS < MIN_SPEECH_MS:
            return np.zeros(0, dtype=np.float32)
        # The trailing quiet carries no words.
        end = len(self.frames) - max(0, self.quiet - 5)
        return np.concatenate(self.frames[:end]).reshape(-1)


def record_utterance(threshold: float) -> tuple[np.ndarray, float | None]:
    """Record until Sir speaks and then pauses.

    The second value is when the speech started, so a follow-up can be timed
    from the moment he begins, not from when the transcript is ready.
    """
    print("Listening, Sir. Speak, then pause.")
    endpoint = Endpointer(threshold)
    started: float | None = None
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME) as stream:
        while True:
            block, _overflowed = stream.read(FRAME)
            done = endpoint.feed(block[:, 0].copy())
            if endpoint.speaking and started is None:
                started = time.monotonic()
            if done:
                break
    return endpoint.audio(), started


def transcribe(ear: WhisperModel, audio: np.ndarray, context: str = "") -> str:
    """Turn one recorded line into text.

    context is the last few lines of the conversation. Whisper reads it as the
    text that came before, so names and words already said are spelled the same way.
    """
    if audio.size == 0:
        return ""
    # beam_size 5 compares a few wordings and keeps the most likely. On the Mac GPU it costs about 0.1s.
    segments, _info = ear.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=f"Friday. {context[-350:]}".strip(),
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


def _speaker_rate(settle: bool) -> int:
    """Use the speaker's rate, after the headphones are back in music mode when that is needed.

    A headset microphone puts them in phone-call mode. Playing during the change
    is what bends the first words. With the Mac's own microphone there is no change to wait for.
    """
    if not settle:
        return int(sd.query_devices(kind="output")["default_samplerate"])
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


def render(voice: PiperVoice | None, text: str, speaker_rate: int) -> np.ndarray | None:
    """Turn one sentence into sound at the speaker's rate. None means nothing to say."""
    if voice is None:
        spoken = text.replace("*", "").replace("#", "").strip()
        if not spoken:
            return None
        try:
            audio, rate = gpt_speech(spoken)
        except OSError as error:
            print(error)
            return None
        return _fit_speaker(audio, rate, speaker_rate, pitch=1.0)

    spoken = for_speech(text)
    if not spoken:
        return None
    # Playing higher raises the pitch and also speeds the audio up.
    # The length scale gives that speed back, so the heard pace stays SPEECH_RATE.
    chunks = list(
        voice.synthesize(
            spoken,
            # Piper's default noise settings give her the natural rise and fall of the sample.
            syn_config=SynthesisConfig(length_scale=PITCH / SPEECH_RATE),
        )
    )
    if not chunks:
        return None
    audio = _finish(
        np.concatenate([chunk.audio_float_array for chunk in chunks]),
        chunks[0].sample_rate,
    )
    if audio.size == 0:
        return None
    return _fit_speaker(audio, chunks[0].sample_rate, speaker_rate, PITCH)


class Mouth:
    """Speak sentences in order on a second thread.

    The brain keeps writing the next sentence while this one plays.
    The next sentence is turned into sound while the current one is still playing.
    """

    def __init__(self, voice: PiperVoice | None, settle: bool) -> None:
        self.voice = voice
        self.settle = settle
        self.lines: queue.Queue[str] = queue.Queue()
        self.rate: int | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, sentence: str) -> None:
        print(f"Friday: {sentence}")
        self.lines.put(sentence)

    def finish(self) -> None:
        """Wait until the last sentence has played, before the microphone listens again."""
        self.lines.join()
        sd.wait()
        self.rate = None

    def _run(self) -> None:
        while True:
            sentence = self.lines.get()
            try:
                if self.rate is None:
                    self.rate = _speaker_rate(self.settle)
                audio = render(self.voice, sentence, self.rate)
                sd.wait()
                if audio is not None:
                    sd.play(audio, self.rate)
            except Exception as error:  # noqa: BLE001
                print(f"Could not speak: {error}")
            finally:
                self.lines.task_done()


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
    # This Mac has a GPU. The spare laptop does not, so it stays on the CPU.
    device = "mps" if sys.platform == "darwin" else "cpu"
    where = "the Mac GPU" if device == "mps" else "the CPU"
    print(f"Loading the ear ({EAR_MODEL}) on {where}. The first run downloads it.")
    try:
        ear = WhisperModel(EAR_MODEL, device=device, compute_type="int8")
    except (RuntimeError, ValueError):
        if device == "cpu":
            raise
        print("The Mac GPU ear is not available. Using the CPU.")
        ear = WhisperModel(EAR_MODEL, device="cpu", compute_type="int8")
    print("Waking the brain.")
    if not warm():
        print("Friday cannot reach Ollama at http://127.0.0.1:11434. Start Ollama, then run this again.")
        return
    mac_mic = use_mac_microphone()
    mouth = Mouth(voice, settle=not mac_mic)
    threshold = room_threshold()
    messages = [{"role": "system", "content": system_prompt()}]
    pending: str | None = None
    addressed_until = 0.0
    print("Say Friday, then the request. After she answers, keep talking without the name.")
    print(f"{FOLLOW_SECONDS} seconds of silence ends the conversation. Ctrl+C stops immediately.")

    while True:
        try:
            audio, started = record_utterance(threshold)
            heard = transcribe(ear, audio, _recent(messages))
        except KeyboardInterrupt:
            print()
            break
        if not heard:
            if audio.size:
                print(f"I caught {audio.size / SAMPLE_RATE:.1f}s of sound but no words.")
            else:
                print("I heard nothing.")
            continue
        print(f"Sir: {heard}")
        began = started if started is not None else time.monotonic()
        in_conversation = began < addressed_until
        named = request_for_friday(heard)

        if pending:
            answer = named if named is not None else heard
            _reply, pending = handle_turn(messages, answer, pending, mouth.say)
            mouth.finish()
            addressed_until = time.monotonic() + FOLLOW_SECONDS
            continue
        if named is None and not in_conversation:
            print("No name in that line. Staying quiet.")
            continue
        request = named if named is not None else heard.strip(" ,.?!")
        if request.lower() in STOP_WORDS:
            mouth.say("Goodbye, Sir.")
            mouth.finish()
            break
        if not request:
            mouth.say("Yes, Sir?")
        else:
            _reply, pending = handle_turn(messages, request, None, mouth.say)
        mouth.finish()
        # The window opens when she stops talking, so her own reply does not use it up.
        addressed_until = time.monotonic() + FOLLOW_SECONDS


def _recent(messages: list[dict]) -> str:
    """The last exchange, as plain text for the ear."""
    lines = [m["content"] for m in messages[1:][-4:] if m["role"] in ("user", "assistant")]
    return " ".join(lines)
