"""Text-to-Speech synthesis using Qwen3-TTS via OpenAI-compatible API."""

import io
import logging

import httpx
import librosa
import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


class Qwen3TTS:
    """Synthesises speech from Chinese text via Qwen3-TTS.

    Produces float32 audio resampled to 16 kHz for the robot speaker.
    """

    def __init__(self) -> None:
        self.payload = {
            "model": config.TTS_MODEL,
            "voice": config.TTS_VOICE,
            "response_format": "wav",
            "instructions": config.TTS_INSTRUCTIONS,
            "task_type": "VoiceDesign",
            "language": "Auto",
        }
        self.api_url = f"{config.TTS_BASE_URL}/v1/audio/speech"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.TTS_API_KEY}",
        }
        log.info("TTS initialised: model=%s voice=%s", config.TTS_MODEL, config.TTS_VOICE)

    def decode_and_resample(
        self, audio_bytes: bytes, target_sr: int = TARGET_SAMPLE_RATE
    ) -> tuple[np.ndarray, int]:
        """Decode WAV bytes and resample to *target_sr* Hz."""
        audio_data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        log.debug("Raw audio: sr=%d shape=%s", samplerate, audio_data.shape)

        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        if samplerate != target_sr:
            audio_data = librosa.resample(
                audio_data, orig_sr=samplerate, target_sr=target_sr
            )

        audio_data = audio_data.reshape(-1, 1).astype(np.float32)
        log.debug("Resampled: sr=%d shape=%s", target_sr, audio_data.shape)
        return audio_data, target_sr

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesise speech for the given Chinese text.

        Returns float32 mono audio at TARGET_SAMPLE_RATE Hz, or an empty array
        on failure.
        """
        self.payload["input"] = text
        with httpx.Client(timeout=300.0) as client:
            response = client.post(self.api_url, json=self.payload, headers=self.headers)

        if response.status_code != 200:
            log.error("TTS error %d: %s", response.status_code, response.text)
            return np.array([], dtype=np.float32)

        try:
            decoded = response.content.decode("utf-8")
            if decoded.startswith('{"error"'):
                log.error("TTS error response: %s", decoded)
                return np.array([], dtype=np.float32)
        except UnicodeDecodeError:
            pass  # Binary audio data, not an error

        audio, _ = self.decode_and_resample(response.content, target_sr=TARGET_SAMPLE_RATE)
        return audio
