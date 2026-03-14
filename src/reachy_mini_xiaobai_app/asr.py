"""Automatic Speech Recognition using Qwen3-ASR via OpenAI-compatible API."""

import io

import numpy as np
import soundfile as sf
from openai import OpenAI

from . import config


class Qwen3ASR:
    """Transcribes Mandarin Chinese audio to text using Qwen3-ASR."""

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=config.ASR_BASE_URL,
            api_key="not-needed",
        )

    def numpy_to_wav_bytes(self, audio_np: np.ndarray, samplerate: int) -> bytes:
        """Convert a NumPy audio array to WAV bytes."""
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()

    def transcribe_audio(self, audio_np: np.ndarray, samplerate: int) -> str:
        """Transcribe a mono float32 audio array to Chinese text."""
        wav_bytes = self.numpy_to_wav_bytes(audio_np, samplerate)
        transcription = self.client.audio.transcriptions.create(
            model=config.ASR_MODEL,
            file=("audio.wav", wav_bytes, "audio/wav"),
        )
        return transcription.text
