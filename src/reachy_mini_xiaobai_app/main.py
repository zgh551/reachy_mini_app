"""Reachy Mini Xiaobai App — main entry point.

Chinese voice conversation loop with a parallel pipeline:

    VAD thread ──→ [vad_to_asr_queue] ──→ ASR thread ──→ [asr_queue] ──→ LLM thread ──→ [tts_queue] ──→ TTS thread ──→ [audio_queue] ──→ Audio thread

Conversation is activated by the keyword "小白" and stays active until
5 seconds of continuous silence (tracked by VAD) after the last audio
finishes playing.  Saying "小白" again while a response is playing
interrupts the current output and starts a new response.

A background MovementExecutor drives the robot's head and antennas via
a separate motion_queue populated by the LLM thread.
"""

import logging
import queue
import re
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
SENTENCE_ENDS = set("。！？.!?")


KEYWORD = "小白"


def _find_sentence_end(text: str) -> int:
    """Return the index of the first sentence-ending punctuation, or -1."""
    for i, ch in enumerate(text):
        if ch in SENTENCE_ENDS:
            return i
    return -1


def _flush_queue(q: queue.Queue) -> None:
    """Drain all items from a queue without blocking."""
    while True:
        try:
            q.get_nowait()
            q.task_done()
        except queue.Empty:
            break


# ---------------------------------------------------------------------------
# Shared conversation state — accessed by all worker threads
# ---------------------------------------------------------------------------


class ChatSession:
    """Thread-safe shared state for the conversation pipeline.

    States and sub-states:
        IDLE                — waiting for keyword activation
        CHATTING/responding — pipeline is producing a response (LLM/TTS/audio)
        CHATTING/waiting    — response finished, waiting for user to speak

    While responding, only the keyword can trigger a barge-in.
    While waiting, any speech continues the conversation.
    After 5 seconds of continuous silence (tracked by VAD thread) we return
    to IDLE.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: str = "IDLE"
        self.responding: bool = False
        self.response_id: int = 0

    def activate(self) -> int:
        """Enter CHATTING/responding state and bump response_id. Returns new id."""
        with self._lock:
            if self.state == "IDLE":
                log.info("Chat activated")
            self.state = "CHATTING"
            self.responding = True
            self.response_id += 1
            return self.response_id

    def deactivate(self) -> None:
        with self._lock:
            if self.state == "CHATTING":
                log.info("Chat deactivated (silence timeout)")
            self.state = "IDLE"
            self.responding = False

    def is_chatting(self) -> bool:
        with self._lock:
            return self.state == "CHATTING"

    def is_responding(self) -> bool:
        with self._lock:
            return self.state == "CHATTING" and self.responding

    def is_waiting(self) -> bool:
        with self._lock:
            return self.state == "CHATTING" and not self.responding

    def get_response_id(self) -> int:
        with self._lock:
            return self.response_id

    def mark_response_done(self) -> None:
        """Called by audio worker when _END_TAG is processed."""
        with self._lock:
            self.responding = False
            log.debug("Response done, VAD will start monitoring for silence")


# Tagged item types for the pipeline queues.
# Each item carries a response_id so workers can discard stale data.
_END_TAG = "END"


# ---------------------------------------------------------------------------
# Worker functions — each runs in its own daemon thread
# ---------------------------------------------------------------------------

def _vad_worker(
    media,
    stop_event: threading.Event,
    session: ChatSession,
    vad_to_asr_queue: queue.Queue,
    llm_client: LLMClient,
) -> None:
    """Capture audio, run VAD tracking 5s silence, push audio to vad_to_asr_queue."""
    vad = VADStateMachine()
    
    # Wait for microphone to be ready
    log.info("Waiting for microphone…")
    while (
        media.get_audio_sample() is None
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
    last_speech_time = time.time()

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

            if vad.state == "SPEAKING":
                last_speech_time = current_time
            elif session.is_waiting():
                # If we are waiting for user input, and there is no speech, track silence
                if current_time - last_speech_time >= 5.0:
                    log.info("VAD detected 5 seconds of silence. Exiting dialogue state.")
                    session.deactivate()
                    llm_client.reset_history()
                    last_speech_time = current_time # Reset to avoid spamming log
            else:
                last_speech_time = current_time

            if result is not None:
                log.info("VAD utterance captured.")
                vad_to_asr_queue.put(result)


def _asr_worker(
    stop_event: threading.Event,
    session: ChatSession,
    vad_to_asr_queue: queue.Queue,
    asr_queue: queue.Queue,
    tts_queue: queue.Queue,
    audio_queue: queue.Queue,
    llm_client: LLMClient,
    media,
) -> None:
    """Read audio from vad_to_asr_queue, run ASR, manage chat state, push text to asr_queue."""
    asr = Qwen3ASR()

    while not stop_event.is_set():
        try:
            audio_array = vad_to_asr_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        log.info("Transcribing…")
        text = asr.transcribe_audio(audio_array, SAMPLE_RATE)
        log.info("ASR result: %s", text)

        has_keyword = KEYWORD in text

        if session.is_responding():
            # Pipeline is busy producing a response.
            if has_keyword:
                # Barge-in: interrupt current response
                log.info("Barge-in detected, interrupting current response")
                _flush_queue(tts_queue)
                _flush_queue(audio_queue)
                try:
                    media.clear_player()
                except Exception:
                    log.debug("clear_player not available, skipping")
                rid = session.activate()
                asr_queue.put((rid, text))
            else:
                log.debug("Ignoring speech while responding")
        elif session.is_waiting():
            # Response finished, waiting for user. Any speech continues.
            log.info("Continuing conversation")
            rid = session.activate()
            asr_queue.put((rid, text))
        elif has_keyword:
            # IDLE → first activation
            llm_client.reset_history()
            rid = session.activate()
            asr_queue.put((rid, text))
        # else: IDLE and no keyword → ignore
        
        vad_to_asr_queue.task_done()


def _llm_worker(
    stop_event: threading.Event,
    session: ChatSession,
    asr_queue: queue.Queue,
    tts_queue: queue.Queue,
    motion_queue: queue.Queue,
    llm_client: LLMClient,
) -> None:
    """Read user text from asr_queue, stream LLM, split into sentences, push to tts_queue."""
    while not stop_event.is_set():
        try:
            item = asr_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        rid, text = item
        sentence_buf = ""
        try:
            for token in llm_client.stream_response(text, motion_queue):
                # Abort if this response has been superseded
                if session.get_response_id() != rid:
                    log.info("LLM response %d superseded, aborting", rid)
                    break

                sentence_buf += token

                while True:
                    end_idx = _find_sentence_end(sentence_buf)
                    if end_idx == -1:
                        break
                    sentence = sentence_buf[: end_idx + 1].strip()
                    sentence_buf = sentence_buf[end_idx + 1 :]

                    if sentence:
                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if clean_sentence:
                            log.info("LLM sentence [%d]: %s", rid, clean_sentence)
                            tts_queue.put((rid, clean_sentence))

            # Flush remaining text (only if not superseded)
            if session.get_response_id() == rid and sentence_buf.strip():
                clean_sentence = re.sub(r'\s+', ' ', sentence_buf).strip()
                if clean_sentence:
                    log.info("LLM sentence (flush) [%d]: %s", rid, clean_sentence)
                    tts_queue.put((rid, clean_sentence))
        except Exception:
            log.exception("Error in LLM streaming")
        finally:
            tts_queue.put((rid, _END_TAG))
            asr_queue.task_done()


def _tts_worker(
    stop_event: threading.Event,
    session: ChatSession,
    tts_queue: queue.Queue,
    audio_queue: queue.Queue,
) -> None:
    """Read sentences from tts_queue, synthesise audio, push to audio_queue."""
    tts = Qwen3TTS()

    while not stop_event.is_set():
        try:
            item = tts_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        rid, payload = item
        try:
            # Discard stale items
            if session.get_response_id() != rid:
                continue

            if payload is _END_TAG:
                audio_queue.put((rid, _END_TAG))
                continue

            log.info("TTS [%d]: %s", rid, payload)
            audio_out = tts.synthesize(payload)
            if len(audio_out) > 0:
                audio_queue.put((rid, audio_out))
        except Exception:
            log.exception("Error in TTS synthesis")
        finally:
            tts_queue.task_done()


def _audio_worker(
    media,
    stop_event: threading.Event,
    session: ChatSession,
    audio_queue: queue.Queue,
) -> None:
    """Read audio arrays from audio_queue and play them on the speaker."""
    samplerate = media.get_output_audio_samplerate()

    while not stop_event.is_set():
        try:
            item = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        rid, payload = item
        try:
            # Discard stale items
            if session.get_response_id() != rid:
                continue

            if payload is _END_TAG:
                time.sleep(0.5)
                session.mark_response_done()
                continue

            media.push_audio_sample(payload)
            time.sleep(len(payload) / samplerate)
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
        vad_to_asr_queue: queue.Queue = queue.Queue()
        asr_queue: queue.Queue = queue.Queue()
        tts_queue: queue.Queue = queue.Queue(maxsize=5)
        audio_queue: queue.Queue = queue.Queue(maxsize=5)
        motion_queue: queue.Queue[dict] = queue.Queue()

        session = ChatSession()
        llm_client = LLMClient()

        media = reachy_mini.media
        media.start_recording()
        media.start_playing()

        # Warm up the playback pipeline by pushing a short silent buffer.
        silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
        media.push_audio_sample(silence)
        time.sleep(0.2)

        # Start all workers
        executor = MovementExecutor(reachy_mini, motion_queue, stop_event)
        executor.start()

        threads = [
            threading.Thread(
                target=_vad_worker,
                args=(media, stop_event, session, vad_to_asr_queue, llm_client),
                daemon=True,
                name="vad",
            ),
            threading.Thread(
                target=_asr_worker,
                args=(stop_event, session, vad_to_asr_queue, asr_queue, tts_queue,
                      audio_queue, llm_client, media),
                daemon=True,
                name="asr",
            ),
            threading.Thread(
                target=_llm_worker,
                args=(stop_event, session, asr_queue, tts_queue,
                      motion_queue, llm_client),
                daemon=True,
                name="llm",
            ),
            threading.Thread(
                target=_tts_worker,
                args=(stop_event, session, tts_queue, audio_queue),
                daemon=True,
                name="tts",
            ),
            threading.Thread(
                target=_audio_worker,
                args=(media, stop_event, session, audio_queue),
                daemon=True,
                name="audio",
            ),
        ]

        for t in threads:
            t.start()

        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=1.0)
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
