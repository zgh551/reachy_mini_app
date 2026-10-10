"""Live test of BreezeTTS against the real service.

Mirrors the synthesize() logic from
src/reachy_mini_xiaobai_app/tts.py but uses scipy for resampling so the test
does not need librosa. Verifies: health check, streaming PCM, header parsing,
normalisation, and WAV output.
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

BASE_URL = "http://192.168.200.252:7860"
TARGET_SAMPLE_RATE = 16000

import httpx  # noqa: E402


def wait_for_service(base_url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return True
            status = response.text.strip()
        except (httpx.HTTPError, ValueError) as exc:
            status = f"unreachable ({exc})"
        if time.monotonic() >= deadline:
            print(f"service not ready: {status}")
            return False
        print(f"waiting for service ({status})...")
        time.sleep(5.0)


def synthesize(text: str, instruction: str = "Speak clearly and naturally.") -> tuple:
    data = {
        "text": text,
        "instruction": instruction,
        "cfg_scale": "1.0",
        "seed": "42",
    }
    started = time.perf_counter()
    chunks: list[bytes] = []
    total = 0
    sample_rate = None
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        with client.stream("POST", f"{BASE_URL}/v1/audio/speech", data=data) as response:
            print(f"HTTP {response.status_code}")
            if response.status_code != 200:
                print(response.read().decode("utf-8", "replace")[:300])
                return None, None
            fmt = response.headers.get("X-Sample-Format", "s16le")
            sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
            print(f"format={fmt} sample_rate={sample_rate}")
            assert fmt == "s16le", f"unexpected format {fmt}"
            first_at = None
            for chunk in response.iter_bytes(16384):
                if chunk:
                    if first_at is None:
                        first_at = time.perf_counter() - started
                    chunks.append(chunk)
                    total += len(chunk)
    print(f"time to first audio: {first_at:.2f}s, total {total} bytes "
          f"in {time.perf_counter() - started:.2f}s")
    raw = b"".join(chunks)
    if len(raw) % 2:
        raw = raw[:-1]
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    # resample 24k -> 16k like the app does (librosa uses soxr; scipy fallback)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(TARGET_SAMPLE_RATE, sample_rate)
    pcm = resample_poly(pcm, TARGET_SAMPLE_RATE // g, sample_rate // g)
    return pcm.reshape(-1, 1).astype(np.float32), sample_rate


def main() -> None:
    assert wait_for_service(BASE_URL), "service not ready"
    print("health OK")

    text = "你好，我是小白，很高兴认识你！"
    audio, orig_sr = synthesize(text)
    if audio is None:
        sys.exit(1)

    dur_orig = len(audio) / TARGET_SAMPLE_RATE
    print(f"audio shape={audio.shape} dtype={audio.dtype} "
          f"peak={np.abs(audio).max():.4f} rms={float(np.sqrt((audio ** 2).mean())):.4f}")
    assert audio.dtype == np.float32 and audio.ndim == 2 and audio.shape[1] == 1
    assert 0.01 < np.abs(audio).max() <= 1.0, "audio is silent or clipped"
    assert dur_orig > 0.5, f"audio too short: {dur_orig:.2f}s"

    out = Path(__file__).parent / "breeze_test_output.wav"
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes((audio[:, 0] * 32767).astype("<i2").tobytes())
    print(f"saved {out} ({dur_orig:.2f}s of speech @ {TARGET_SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
