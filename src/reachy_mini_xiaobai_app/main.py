"""Reachy Mini Xiaobai App — main entry point.

Chinese voice conversation loop with a 4-stage parallel pipeline:

    ASR thread ──→ [asr_queue] ──→ LLM thread ──→ [tts_queue] ──→ TTS thread ──→ [audio_queue] ──→ Audio thread

A background MovementExecutor drives the robot's head and antennas via
a separate motion_queue populated by the LLM thread.
"""

import logging
import queue
import threading
import time

import numpy as np
import numpy.typing as npt

from reachy_mini import ReachyMini

from .asr import Qwen3ASR
from .llm import LLMClient
from .moves import MovementExecutor
from .tts import Qwen3TTS
from .vad import VADStateMachine

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512
TIMEOUT = 10
SENTENCE_ENDS = set("。！？.!?")

# Sentinel object to signal end-of-response across queues.
_END = None


def _find_sentence_end(text: str) -> int:
    """Return the index of the first sentence-ending punctuation, or -1."""
    for i, ch in enumerate(text):
        if ch in SENTENCE_ENDS:
            return i
    return -1


# ---------------------------------------------------------------------------
# Worker functions — each runs in its own daemon thread
# ---------------------------------------------------------------------------


def _asr_worker(
    media,
    stop_event: threading.Event,
    asr_queue: "queue.Queue[str | None]",
) -> None:
    """Capture audio from the microphone, run VAD + ASR, push text to asr_queue."""
    vad = VADStateMachine()
    asr = Qwen3ASR()

    # Wait for microphone to be ready
    log.info("Waiting for microphone…")
    start_time = time.time()
    while (
        media.get_audio_sample() is None
        and time.time() - start_time < TIMEOUT
        and not stop_event.is_set()
    ):
        time.sleep(0.005)

    samplerate = media.get_input_audio_samplerate()
    log.info("Microphone ready, sample rate: %d Hz", samplerate)

    need_resample = samplerate != SAMPLE_RATE
    if need_resample:
        import librosa  # noqa: F401

        log.info("Will resample from %d Hz to %d Hz", samplerate, SAMPLE_RATE)

    audio_accumulator = np.array([], dtype=np.float32)

    while not stop_event.is_set():
        sample = media.get_audio_sample()
        if sample is None:
            time.sleep(0.005)
            continue

        # Stereo → mono
        if sample.ndim == 2:
            mono = sample.mean(axis=1).astype(np.float32)
        else:
            mono = sample.astype(np.float32)

        if need_resample:
            mono = librosa.resample(mono, orig_sr=samplerate, target_sr=SAMPLE_RATE)

        audio_accumulator = np.concatenate([audio_accumulator, mono])

        while len(audio_accumulator) >= VAD_CHUNK_SIZE:
            vad_chunk = audio_accumulator[:VAD_CHUNK_SIZE]
            audio_accumulator = audio_accumulator[VAD_CHUNK_SIZE:]

            current_time = time.time()
            result = vad.process_chunk(vad_chunk, current_time)

            if result is not None:
                log.info("Transcribing…")
                text = asr.transcribe_audio(result, SAMPLE_RATE)
                log.info("ASR result: %s", text)

                if "小白" in text:
                    asr_queue.put(text)


def _llm_worker(
    stop_event: threading.Event,
    asr_queue: "queue.Queue[str | None]",
    tts_queue: "queue.Queue[str | None]",
    motion_queue: "queue.Queue[dict]",
) -> None:
    """Read user text from asr_queue, stream LLM, split into sentences, push to tts_queue."""
    llm = LLMClient()

    while not stop_event.is_set():
        try:
            text = asr_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        sentence_buf = ""
        try:
            for token in llm.stream_response(text, motion_queue):
                sentence_buf += token

                while True:
                    end_idx = _find_sentence_end(sentence_buf)
                    if end_idx == -1:
                        break
                    sentence = sentence_buf[: end_idx + 1].strip()
                    sentence_buf = sentence_buf[end_idx + 1 :]

                    if sentence:
                        log.info("LLM sentence: %s", sentence)
                        tts_queue.put(sentence)

            # Flush remaining text
            if sentence_buf.strip():
                log.info("LLM sentence (flush): %s", sentence_buf.strip())
                tts_queue.put(sentence_buf.strip())
        except Exception:
            log.exception("Error in LLM streaming")
        finally:
            tts_queue.put(_END)
            asr_queue.task_done()


def _tts_worker(
    stop_event: threading.Event,
    tts_queue: "queue.Queue[str | None]",
    audio_queue: "queue.Queue[npt.NDArray[np.float32] | None]",
) -> None:
    """Read sentences from tts_queue, synthesise audio, push to audio_queue."""
    tts = Qwen3TTS()

    while not stop_event.is_set():
        try:
            sentence = tts_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            if sentence is _END:
                audio_queue.put(_END)
                continue

            log.info("TTS: %s", sentence)
            audio_out = tts.synthesize(sentence)
            if len(audio_out) > 0:
                audio_queue.put(audio_out)
        except Exception:
            log.exception("Error in TTS synthesis")
        finally:
            tts_queue.task_done()


def _audio_worker(
    media,
    stop_event: threading.Event,
    audio_queue: "queue.Queue[npt.NDArray[np.float32] | None]",
) -> None:
    """Read audio arrays from audio_queue and play them on the speaker."""
    samplerate = media.get_output_audio_samplerate()

    while not stop_event.is_set():
        try:
            audio = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            if audio is _END:
                time.sleep(0.5)
                continue

            media.push_audio_sample(audio)
            time.sleep(len(audio) / samplerate)
        except Exception:
            log.exception("Error in audio playback")
        finally:
            audio_queue.task_done()


# ---------------------------------------------------------------------------
# Application class
# ---------------------------------------------------------------------------


class ReachyMiniXiaobaiApp:
    """Voice-conversation app for Reachy Mini with LLM-driven motions."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Start all pipeline threads and wait for stop_event."""
        # Inter-stage queues
        asr_queue: queue.Queue[str | None] = queue.Queue()
        tts_queue: queue.Queue[str | None] = queue.Queue(maxsize=5)
        audio_queue: queue.Queue[npt.NDArray[np.float32] | None] = queue.Queue(maxsize=5)
        motion_queue: queue.Queue[dict] = queue.Queue()

        media = reachy_mini.media
        media.start_recording()
        media.start_playing()

        # Start all workers
        executor = MovementExecutor(reachy_mini, motion_queue, stop_event)
        executor.start()

        threads = [
            threading.Thread(
                target=_asr_worker,
                args=(media, stop_event, asr_queue),
                daemon=True,
                name="asr",
            ),
            threading.Thread(
                target=_llm_worker,
                args=(stop_event, asr_queue, tts_queue, motion_queue),
                daemon=True,
                name="llm",
            ),
            threading.Thread(
                target=_tts_worker,
                args=(stop_event, tts_queue, audio_queue),
                daemon=True,
                name="tts",
            ),
            threading.Thread(
                target=_audio_worker,
                args=(media, stop_event, audio_queue),
                daemon=True,
                name="audio",
            ),
        ]

        for t in threads:
            t.start()

        try:
            # Block until shutdown is requested
            stop_event.wait()
        finally:
            media.stop_playing()
            media.stop_recording()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = ReachyMiniXiaobaiApp()
    stop_event = threading.Event()

    with ReachyMini() as mini:
        try:
            app.run(mini, stop_event)
        except KeyboardInterrupt:
            log.info("Shutting down…")
            stop_event.set()


if __name__ == "__main__":
    main()
