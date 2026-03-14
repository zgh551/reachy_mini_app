"""Reachy Mini Xiaobai App — main entry point.

Chinese voice conversation loop: VAD → ASR → LLM (with tool calls) → TTS,
with a background MovementExecutor that drives the robot's head and antennas.
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


def _find_sentence_end(text: str) -> int:
    """Return the index of the first sentence-ending punctuation, or -1."""
    for i, ch in enumerate(text):
        if ch in SENTENCE_ENDS:
            return i
    return -1


def _push_audio(media, audio: npt.NDArray[np.float32]) -> None:
    """Push audio to the speaker."""
    media.push_audio_sample(audio)


class ReachyMiniXiaobaiApp:
    """Voice-conversation app for Reachy Mini with LLM-driven motions."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Main application loop called by the Reachy Mini framework."""
        vad = VADStateMachine()
        asr = Qwen3ASR()
        llm = LLMClient()
        tts = Qwen3TTS()
        motion_queue: queue.Queue[dict] = queue.Queue()

        # Start background motion executor
        executor = MovementExecutor(reachy_mini, motion_queue, stop_event)
        executor.start()

        reachy_mini.media.start_recording()

        # Wait for microphone
        log.info("Waiting for microphone…")
        start_time = time.time()
        while (
            reachy_mini.media.get_audio_sample() is None
            and time.time() - start_time < TIMEOUT
            and not stop_event.is_set()
        ):
            time.sleep(0.005)

        samplerate = reachy_mini.media.get_input_audio_samplerate()
        log.info("Microphone ready, sample rate: %d Hz", samplerate)

        need_resample = samplerate != SAMPLE_RATE
        if need_resample:
            import librosa  # noqa: F401 — lazy import, only when needed
            log.info("Will resample from %d Hz to %d Hz", samplerate, SAMPLE_RATE)

        audio_accumulator = np.array([], dtype=np.float32)

        try:
            while not stop_event.is_set():
                sample = reachy_mini.media.get_audio_sample()
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
                            self._respond(reachy_mini, llm, tts, text, motion_queue)
        finally:
            reachy_mini.media.stop_recording()

    def _respond(
        self,
        mini: ReachyMini,
        llm: LLMClient,
        tts: Qwen3TTS,
        text: str,
        motion_queue: "queue.Queue[dict]",
    ) -> None:
        """Stream an LLM response, synthesise speech sentence-by-sentence."""
        sentence_buf = ""
        mini.media.start_playing()
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
                        log.info("TTS: %s", sentence)
                        audio_out = tts.synthesize(sentence)
                        if len(audio_out) > 0:
                            _push_audio(mini.media, audio_out)
                            time.sleep(
                                len(audio_out) / mini.media.get_output_audio_samplerate()
                            )

            # Flush remaining text
            if sentence_buf.strip():
                log.info("TTS (flush): %s", sentence_buf.strip())
                audio_out = tts.synthesize(sentence_buf.strip())
                if len(audio_out) > 0:
                    _push_audio(mini.media, audio_out)
                    time.sleep(
                        len(audio_out) / mini.media.get_output_audio_samplerate()
                    )
        finally:
            time.sleep(0.5)
            mini.media.stop_playing()


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
