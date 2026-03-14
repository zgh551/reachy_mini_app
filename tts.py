import base64
import os
import httpx
import io
import numpy as np
import soundfile as sf
import librosa
import time

class Qwen3TTS:
    """Text-to-speech synthesiser using Kokoro (ONNX, local, no cloud).

    Produces float32 audio at 24 kHz which can then be resampled to match
    the robot's speaker sample rate (16 kHz).
    """

    # Kokoro native output sample rate
    KOKORO_SAMPLE_RATE = 24000
    # Robot speaker sample rate (from AudioBase.SAMPLE_RATE)
    TARGET_SAMPLE_RATE = 16000

    # Default server configuration
    DEFAULT_API_BASE = "http://192.168.200.252:8091"
    DEFAULT_API_KEY = "no-needed"

    def __init__(self, lang: str = "z") -> None:
        """Load the Kokoro ONNX model.

        Args:
            lang: Language code passed to Kokoro.  Use "z" for Chinese
                  (Mandarin) voices.
        """
        """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""

        # Build request payload
        self.payload = {
            "model": "./Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "voice": "vivian",
            "response_format": "wav",
        }

        # Add optional parameters
        self.payload["instructions"] = "你是赛车总动员中的闪电麦昆，语言欢快，性格阳光"
        self.payload["task_type"] = "VoiceDesign"
        self.payload["language"] = "Auto"
        # self.payload["max_new_tokens"] = 8192

        print(self.payload)
        print("Generating audio...")
        # Make the API call
        self.api_url = f"{self.DEFAULT_API_BASE}/v1/audio/speech"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.DEFAULT_API_KEY}",
        }

    def decode_and_resample(self, audio_bytes, target_sr=16000):
        """解码音频 bytes 并重采样到 16kHz，输出符合 push_audio_sample 要求的格式"""
        
        # 1. 解码 WAV bytes → numpy
        audio_data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        print(f"原始: sr={samplerate}, shape={audio_data.shape}")
        
        # 2. 如果是多声道，先转单声道（方便重采样）
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # (samples,)
        
        # 3. 重采样 12kHz → 16kHz
        if samplerate != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=target_sr)
        
        # 4. 关键：reshape 为 (samples, 1)，满足 push_audio_sample 要求
        audio_data = audio_data.reshape(-1, 1).astype(np.float32)
        
        print(f"输出: sr={target_sr}, shape={audio_data.shape}, dtype={audio_data.dtype}")
        return audio_data, target_sr


    def synthesize(self, text: str):
        """Synthesise speech for the given Chinese text.

        Args:
            text: Chinese text to convert to speech.

        Returns:
            Float32 mono audio array at TARGET_SAMPLE_RATE Hz, values in [-1, 1].
            Returns an empty array for blank input.
        """
        self.payload["input"] = text
        with httpx.Client(timeout=300.0) as client:
            response = client.post(self.api_url, json=self.payload, headers=self.headers)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return

        # Check for JSON error response (only if content is valid UTF-8 text)
        try:
            text = response.content.decode("utf-8")
            if text.startswith('{"error"'):
                print(f"Error: {text}")
                return
        except UnicodeDecodeError:
            pass  # Binary audio data, not an error
        audio, _ = self.decode_and_resample(response.content, target_sr=self.TARGET_SAMPLE_RATE)

        # audio = np.asarray(samples, dtype=np.float32)

        # # Resample from Kokoro rate to robot speaker rate if needed
        # if sample_rate != self.TARGET_SAMPLE_RATE:
        #     audio = _resample(audio, sample_rate, self.TARGET_SAMPLE_RATE)

        return audio

