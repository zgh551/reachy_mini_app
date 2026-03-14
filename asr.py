import base64
import httpx
import io
from openai import OpenAI
import soundfile as sf

# Initialize client
# client = OpenAI(
#     base_url="http://192.168.1.18:8000/v1",
#     api_key="EMPTY"
# )
ASR_BASE_URL="http://192.168.1.18:8000/v1"
MODEL_NAME="./Qwen3-ASR-1.7B"

class Qwen3ASR:
    """Automatic speech recogniser wrapping Qwe3-ASR.

    Transcribes Mandarin Chinese audio to text using the qwen3-asr model.
    """

    def __init__(self, model_size: str = "medium") -> None:
        """Load the Qwen3-ASR model.

        Args:
            model_size: Whisper model variant.  "medium" balances quality
                        and speed for Chinese (~1.5 GB).
        """
        # ============ 初始化 ============
        self.client = OpenAI(
            base_url=ASR_BASE_URL,
            api_key="not-needed",
        )

    def numpy_to_wav_bytes(self, audio_np, samplerate):
        """NumPy → WAV bytes"""
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()


    def transcribe_audio(self, audio_np, samplerate):
        """Transcribe a mono float32 audio array to Chinese text.

        Args:
            audio: Float32 mono audio at 16 kHz.

        Returns:
            Transcribed text string (may be empty if no speech detected).
        """
        wav_bytes = self.numpy_to_wav_bytes(audio_np, samplerate)
        transcription = self.client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=("audio.wav", wav_bytes, "audio/wav"),
        )
        return transcription.text