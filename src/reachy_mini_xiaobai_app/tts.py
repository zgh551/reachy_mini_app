"""Text-to-Speech synthesis.

Supports two backends:

- ``Qwen3TTS``: Qwen3-TTS via an OpenAI-compatible ``/v1/audio/speech`` API.
- ``BreezeTTS``: Breeze TTS-2 via its streaming raw-PCM ``/v1/audio/speech``
  API (service started with
  ``python -m breeze_infer.api <model> --host 0.0.0.0 --port 7860``).

Use :func:`create_tts` to build the backend selected by ``config.TTS_PROVIDER``.
"""

import io
import logging
import time
from pathlib import Path

import httpx
import librosa
import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

# Breeze TTS-2 stream constants (raw mono 16-bit little-endian PCM).
BREEZE_SAMPLE_FORMAT = "s16le"
BREEZE_FALLBACK_SAMPLE_RATE = 24000
BREEZE_STREAM_CHUNK_SIZE = 16384


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


class BreezeTTS:
    """Synthesises speech via the Breeze TTS-2 streaming API.

    The ``/v1/audio/speech`` endpoint streams raw mono 16-bit little-endian
    PCM; the sample rate is reported in the ``X-Sample-Rate`` response header
    (``X-Sample-Format`` must be ``s16le``). Audio is resampled to 16 kHz
    float32 for the robot speaker, matching :class:`Qwen3TTS`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        instruction: str | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or config.BREEZE_TTS_BASE_URL).rstrip("/")
        self.instruction = (
            instruction
            if instruction is not None
            else config.BREEZE_TTS_INSTRUCTION
        )
        self.cfg_scale = (
            cfg_scale if cfg_scale is not None else config.BREEZE_TTS_CFG_SCALE
        )
        self.seed = seed if seed is not None else config.BREEZE_TTS_SEED
        self.timeout = timeout if timeout is not None else config.BREEZE_TTS_TIMEOUT

        # Optional voice cloning: only active when both are configured.
        self.ref_audio: Path | None = None
        self.ref_text = ""
        if ref_audio is None:
            if config.BREEZE_TTS_REF_AUDIO and config.BREEZE_TTS_REF_TEXT:
                ref_audio = config.BREEZE_TTS_REF_AUDIO
                ref_text = config.BREEZE_TTS_REF_TEXT
            else:
                ref_audio, ref_text = None, None
        if ref_audio:
            if not ref_text or not ref_text.strip():
                raise ValueError(
                    "BreezeTTS requires ref_text when ref_audio is provided."
                )
            ref_path = Path(ref_audio)
            if not ref_path.is_file():
                raise FileNotFoundError(f"reference audio not found: {ref_path}")
            self.ref_audio = ref_path
            self.ref_text = ref_text
        if self.cfg_scale <= 0:
            raise ValueError("cfg_scale must be greater than 0.")

        self.api_url = f"{self.base_url}/v1/audio/speech"
        log.info(
            "TTS initialised: backend=BreezeTTS-2 base_url=%s cfg_scale=%s seed=%d"
            " ref_audio=%s",
            self.base_url,
            self.cfg_scale,
            self.seed,
            self.ref_audio or "none",
        )

    def wait_for_service(self, timeout_s: float = 600.0) -> bool:
        """Poll ``/health`` until the model is loaded.

        Returns True when ready, False if the service did not become ready
        within *timeout_s*.
        """
        deadline = time.monotonic() + timeout_s
        url = f"{self.base_url}/health"
        while True:
            try:
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    return True
                status = response.text.strip()
            except (httpx.HTTPError, ValueError) as exc:
                status = f"unreachable ({exc})"
            if time.monotonic() >= deadline:
                log.error("Breeze TTS service not ready within %.0fs: %s", timeout_s, status)
                return False
            log.info("Waiting for Breeze TTS service (%s)...", status)
            time.sleep(5.0)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesise speech for the given text.

        Returns float32 mono audio at TARGET_SAMPLE_RATE Hz, or an empty array
        on failure.
        """
        data = {
            "text": text,
            "instruction": self.instruction,
            "cfg_scale": str(self.cfg_scale),
            "seed": str(self.seed),
        }
        files = None
        if self.ref_audio is not None:
            data["ref_text"] = self.ref_text
            files = {"ref_audio": (self.ref_audio.name, open(self.ref_audio, "rb"), "audio/wav")}

        started = time.perf_counter()
        try:
            # stream=True: the server starts streaming PCM as soon as it is
            # generated, so read the body incrementally.
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
                with client.stream(
                    "POST", self.api_url, data=data, files=files
                ) as response:
                    if response.status_code == 409:
                        log.error("Breeze TTS server busy (single concurrency)")
                        return np.array([], dtype=np.float32)
                    if response.status_code != 200:
                        body = response.read().decode("utf-8", errors="replace")
                        log.error(
                            "Breeze TTS error %d: %s", response.status_code, body[:200]
                        )
                        return np.array([], dtype=np.float32)

                    sample_format = response.headers.get(
                        "X-Sample-Format", BREEZE_SAMPLE_FORMAT
                    )
                    if sample_format != BREEZE_SAMPLE_FORMAT:
                        log.error("Unsupported Breeze sample format: %s", sample_format)
                        return np.array([], dtype=np.float32)
                    sample_rate = int(
                        response.headers.get(
                            "X-Sample-Rate", str(BREEZE_FALLBACK_SAMPLE_RATE)
                        )
                    )

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes(BREEZE_STREAM_CHUNK_SIZE):
                        if chunk:
                            chunks.append(chunk)
                            total += len(chunk)
        except httpx.HTTPError as exc:
            log.error("Breeze TTS request failed: %s", exc)
            return np.array([], dtype=np.float32)
        finally:
            if files is not None:
                files["ref_audio"][1].close()

        if total == 0:
            log.error("Breeze TTS returned no audio for text: %r", text[:50])
            return np.array([], dtype=np.float32)
        log.info(
            "Breeze TTS: %d PCM bytes in %.2fs (sr=%d)",
            total,
            time.perf_counter() - started,
            sample_rate,
        )

        raw = b"".join(chunks)
        if len(raw) % 2:  # truncated s16le frame; drop the dangling byte
            log.warning("Breeze TTS stream had odd byte count (%d); dropping last byte", len(raw))
            raw = raw[:-1]
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if sample_rate != TARGET_SAMPLE_RATE:
            pcm = librosa.resample(
                pcm, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
            )
        return pcm.reshape(-1, 1).astype(np.float32)


def create_tts():
    """Build the TTS backend selected by ``config.TTS_PROVIDER``.

    Returns a Qwen3TTS or BreezeTTS instance; raises ValueError for unknown
    providers.
    """
    provider = (config.TTS_PROVIDER or "qwen3").strip().lower()
    if provider in ("qwen3", "qwen", "qwen3-tts"):
        return Qwen3TTS()
    if provider in ("breeze", "breeze-tts", "breeze-tts-2", "breeze2"):
        return BreezeTTS()
    raise ValueError(
        f"Unknown TTS_PROVIDER {config.TTS_PROVIDER!r} (expected 'qwen3' or 'breeze')"
    )
