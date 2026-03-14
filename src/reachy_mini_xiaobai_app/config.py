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

# TTS
TTS_BASE_URL: str = os.getenv("TTS_BASE_URL", "http://192.168.200.252:8091")
TTS_API_KEY: str = os.getenv("TTS_API_KEY", "not-needed")
TTS_MODEL: str = os.getenv("TTS_MODEL", "./Qwen3-TTS-12Hz-1.7B-VoiceDesign")
TTS_VOICE: str = os.getenv("TTS_VOICE", "vivian")
TTS_INSTRUCTIONS: str = os.getenv(
    "TTS_INSTRUCTIONS", "你是赛车总动员中的闪电麦昆，语言欢快，性格阳光"
)
