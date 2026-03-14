"""Voice Activity Detection state machine using Silero VAD."""

import logging

import numpy as np
import torch
from silero_vad import load_silero_vad

log = logging.getLogger(__name__)

# VAD parameters
SILENCE_THRESHOLD = 1.5    # seconds of silence before treating speech as finished
MIN_SPEECH_DURATION = 0.3  # discard segments shorter than this (noise)
MAX_SPEECH_DURATION = 30   # force-flush after this many seconds

SAMPLE_RATE = 16000


def is_speech(audio_chunk: np.ndarray, model, sample_rate: int = 16000) -> float:
    """Return speech probability (0.0–1.0) for a short audio chunk."""
    audio_tensor = torch.from_numpy(audio_chunk).float()
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor[:, 0]
    return model(audio_tensor, sample_rate).item()


class VADStateMachine:
    """Three-state machine: WAITING → SPEAKING → SILENCE.

    Feed fixed-size audio chunks via :meth:`process_chunk`.  When a complete
    utterance is detected the method returns the concatenated audio; otherwise
    it returns ``None``.
    """

    def __init__(self) -> None:
        self.state = "WAITING"
        self.speech_buffer: list[np.ndarray] = []
        self.silence_start_time: float | None = None
        self.speech_start_time: float | None = None
        self.speech_prob_threshold = 0.5
        self.vad_model = load_silero_vad()

    def process_chunk(self, audio_chunk: np.ndarray, current_time: float) -> np.ndarray | None:
        """Process one audio chunk.

        Returns the full utterance as a numpy array when speech ends, else ``None``.
        """
        prob = is_speech(audio_chunk, self.vad_model, SAMPLE_RATE)

        if self.state == "WAITING":
            if prob > self.speech_prob_threshold:
                log.info("Speech detected (prob=%.2f)", prob)
                self.state = "SPEAKING"
                self.speech_start_time = current_time
                self.speech_buffer = [audio_chunk]
            return None

        elif self.state == "SPEAKING":
            self.speech_buffer.append(audio_chunk)
            if prob > self.speech_prob_threshold:
                if current_time - self.speech_start_time > MAX_SPEECH_DURATION:
                    log.warning("Max duration %ss exceeded, flushing", MAX_SPEECH_DURATION)
                    return self._flush()
            else:
                self.state = "SILENCE"
                self.silence_start_time = current_time
            return None

        elif self.state == "SILENCE":
            self.speech_buffer.append(audio_chunk)
            if prob > self.speech_prob_threshold:
                log.debug("Brief pause, speech resumed")
                self.state = "SPEAKING"
                self.silence_start_time = None
                return None

            silence_duration = current_time - self.silence_start_time
            if silence_duration >= SILENCE_THRESHOLD:
                speech_duration = current_time - self.speech_start_time
                log.info(
                    "Silence %.1fs, utterance %.1fs", silence_duration, speech_duration
                )
                if speech_duration - silence_duration < MIN_SPEECH_DURATION:
                    log.debug("Utterance too short, discarding")
                    self._reset()
                    return None
                return self._flush()
            return None

        return None

    def _flush(self) -> np.ndarray:
        audio_data = np.concatenate(self.speech_buffer, axis=0)
        self._reset()
        return audio_data

    def _reset(self) -> None:
        self.state = "WAITING"
        self.speech_buffer = []
        self.silence_start_time = None
        self.speech_start_time = None
        self.vad_model.reset_states()
