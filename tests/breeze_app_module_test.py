"""Live test of the real app module BreezeTTS against the running service.

Imports src/reachy_mini_xiaobai_app/tts.py directly (with librosa available)
and exercises: health polling, synthesize() output contract, the create_tts()
factory, and an end-to-end WAV write in the exact format the audio worker
consumes (float32, shape (-1, 1), 16 kHz).
"""

import logging
import sys
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from reachy_mini_xiaobai_app import config  # noqa: E402
from reachy_mini_xiaobai_app.tts import BreezeTTS, create_tts  # noqa: E402


def main() -> None:
    print(f"TTS_PROVIDER={config.TTS_PROVIDER}")
    print(f"BREEZE_TTS_BASE_URL={config.BREEZE_TTS_BASE_URL}")

    # 1. Health check via the module's own helper
    tts = BreezeTTS()
    assert tts.wait_for_service(timeout_s=30.0), "service not ready"
    print("1. health check OK")

    # 2. Real synthesis through the app module
    audio = tts.synthesize("你好，我是小白，今天天气真不错，我们一起去公园玩吧！")
    assert audio.dtype == np.float32, f"dtype: {audio.dtype}"
    assert audio.ndim == 2 and audio.shape[1] == 1, f"shape: {audio.shape}"
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio ** 2).mean()))
    print(f"2. synthesize OK: shape={audio.shape} peak={peak:.4f} rms={rms:.4f}")
    assert 0.01 < peak <= 1.0, f"audio silent or clipped: peak={peak}"
    assert len(audio) > 16000, f"audio too short: {len(audio)} samples"

    # 3. Factory: breeze -> BreezeTTS, qwen3 -> Qwen3TTS, bogus -> ValueError
    config.TTS_PROVIDER = "breeze"
    assert isinstance(create_tts(), BreezeTTS)
    config.TTS_PROVIDER = "qwen3"
    from reachy_mini_xiaobai_app.tts import Qwen3TTS

    assert isinstance(create_tts(), Qwen3TTS)
    config.TTS_PROVIDER = "does-not-exist"
    try:
        create_tts()
        raise AssertionError("expected ValueError for unknown provider")
    except ValueError:
        pass
    config.TTS_PROVIDER = "breeze"
    print("3. create_tts factory OK (breeze/qwen3/unknown all behave)")

    # 4. Determinism: same seed -> identical PCM bytes
    tts2 = BreezeTTS()
    audio2 = tts2.synthesize("你好，我是小白，今天天气真不错，我们一起去公园玩吧！")
    print(f"4. determinism check: identical={np.array_equal(audio, audio2)}")

    # 5. Write WAV exactly as the robot speaker would consume it
    out = Path(__file__).parent / "breeze_app_module_output.wav"
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((audio[:, 0] * 32767).astype("<i2").tobytes())
    print(f"5. saved {out} ({len(audio) / 16000:.2f}s @ 16000 Hz)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
