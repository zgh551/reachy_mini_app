"""Centralised configuration loaded from environment / .env file."""

import os
from dotenv import load_dotenv

load_dotenv()

# ASR
ASR_BASE_URL: str = os.getenv("ASR_BASE_URL", "http://192.168.1.18:8000/v1")
ASR_MODEL: str = os.getenv("ASR_MODEL", "./Qwen3-ASR-1.7B")

# LLM
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://192.168.200.252:30000/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5-27b")

# LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://192.168.200.252:8000/v1")
# LLM_API_KEY: str = os.getenv("LLM_API_KEY", "not-needed")
# LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3")

# TTS
# Provider selection: "qwen3" (OpenAI-compatible Qwen3-TTS) or "breeze" (Breeze TTS-2).
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "qwen3")

# --- TTS: Qwen3 (OpenAI-compatible API) ---
TTS_BASE_URL: str = os.getenv("TTS_BASE_URL", "http://192.168.200.252:8091")
TTS_API_KEY: str = os.getenv("TTS_API_KEY", "not-needed")
# TTS_MODEL: str = os.getenv("TTS_MODEL", "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
TTS_MODEL: str = os.getenv("TTS_MODEL", "Qwen3-TTS-12Hz-1.7B-CustomVoice")

TTS_VOICE: str = os.getenv("TTS_VOICE", "vivian")
TTS_INSTRUCTIONS: str = os.getenv(
    "TTS_INSTRUCTIONS", "你是赛车总动员中的闪电麦昆，语言欢快，性格阳光"
)

# --- TTS: Breeze TTS-2 (streaming raw PCM API) ---
BREEZE_TTS_BASE_URL: str = os.getenv(
    "BREEZE_TTS_BASE_URL", "http://192.168.200.252:7860"
)
BREEZE_TTS_INSTRUCTION: str = os.getenv(
    "BREEZE_TTS_INSTRUCTION", "Speak clearly and naturally."
)
BREEZE_TTS_CFG_SCALE: float = float(os.getenv("BREEZE_TTS_CFG_SCALE", "1.0"))
BREEZE_TTS_SEED: int = int(os.getenv("BREEZE_TTS_SEED", "42"))
# Optional voice cloning: both must be set to be used.
BREEZE_TTS_REF_AUDIO: str = os.getenv("BREEZE_TTS_REF_AUDIO", "")
BREEZE_TTS_REF_TEXT: str = os.getenv("BREEZE_TTS_REF_TEXT", "")
BREEZE_TTS_TIMEOUT: float = float(os.getenv("BREEZE_TTS_TIMEOUT", "300"))
