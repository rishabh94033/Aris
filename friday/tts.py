"""Speak with gpt-4o-mini-tts, the text-to-speech model behind ChatGPT voice.

The words Friday says are sent to OpenAI when a key is set. The microphone
audio stays on this Mac, and the chat model stays on this Mac.
"""

import io
import json
import os
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEECH_URL = "https://api.openai.com/v1/audio/speech"
MODEL = "gpt-4o-mini-tts"
# cedar and marin are OpenAI's recommended voices. cedar is the calmer of the two.
VOICE = "cedar"
INSTRUCTIONS = (
    "Calm adult assistant speaking to one person in the room. "
    "Keep one steady pitch from the first word to the last. "
    "Warm and clear. Do not brighten or rush the ending."
)


def _load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def gpt_ready() -> bool:
    _load_env()
    return bool(os.environ.get("OPENAI_API_KEY"))


def gpt_speech(text: str) -> tuple[np.ndarray, int]:
    """Return the spoken waveform and its sample rate."""
    _load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    body = json.dumps(
        {
            "model": MODEL,
            "voice": VOICE,
            "input": text,
            "instructions": INSTRUCTIONS,
            "response_format": "wav",
            "speed": 1.0,
        }
    ).encode()
    request = urllib.request.Request(
        SPEECH_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace").strip()
        raise OSError(f"OpenAI speech returned {error.code}. {detail}") from error
    except urllib.error.URLError as error:
        raise OSError(f"Could not reach OpenAI speech. {error.reason}") from error

    with wave.open(io.BytesIO(payload), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio), rate
