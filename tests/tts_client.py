#!/usr/bin/env python3
"""Python client for the Breeze TTS 2 streaming API.

Works with the service started via:

    python -m breeze_infer.api ../breeze-tts-2 --host 0.0.0.0 --port 7860

The API streams raw mono 16-bit little-endian PCM from ``/v1/audio/speech``
(the sample rate is reported in the ``X-Sample-Rate`` response header). This
client consumes the stream and saves it as a WAV file. It requires ``requests``:

    python -m pip install requests
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_INSTRUCTION = "Speak clearly and naturally."
FALLBACK_SAMPLE_RATE = 24000
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate


def wait_for_service(base_url: str, timeout_s: float = 600.0) -> None:
    """Poll /health until the model has finished loading."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
            status = response.text.strip()
        except requests.RequestException:
            status = "unreachable"
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"service not ready within {timeout_s}s (last: {status})"
            )
        print(f"waiting for service ({status})...", flush=True)
        time.sleep(5.0)


def read_wav_info(path: str | Path) -> WavInfo:
    """Read the header of a PCM WAV file."""
    with wave.open(str(path), "rb") as wav:
        return WavInfo(
            sample_rate=wav.getframerate(),
            channels=wav.getnchannels(),
            frames=wav.getnframes(),
        )


def synthesize_speech(
    text: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    instruction: str = DEFAULT_INSTRUCTION,
    cfg_scale: float = 1.0,
    ref_audio: str | Path | None = None,
    ref_text: str = "",
    seed: int = 42,
    output: str | Path = "speech.wav",
) -> Path:
    """Synthesize ``text`` and save the streamed PCM as a 16-bit mono WAV."""
    has_reference = ref_audio is not None
    if has_reference != bool(ref_text.strip()):
        raise ValueError(
            "ref_audio and ref_text must be provided together or both omitted."
        )
    if cfg_scale <= 0:
        raise ValueError("cfg_scale must be greater than 0.")

    data = {
        "text": text,
        "instruction": instruction,
        "cfg_scale": str(cfg_scale),
        "seed": str(seed),
    }
    url = f"{base_url.rstrip('/')}/v1/audio/speech"
    started = time.perf_counter()
    with ExitStack() as stack:
        files = None
        if has_reference:
            ref_path = Path(ref_audio)
            if not ref_path.is_file():
                raise FileNotFoundError(f"reference audio not found: {ref_path}")
            data["ref_text"] = ref_text
            handle = stack.enter_context(open(ref_path, "rb"))
            files = {"ref_audio": (ref_path.name, handle, "audio/wav")}
        try:
            # The request body (including the reference audio) is fully sent
            # before post() returns, so closing the handle here is safe.
            response = requests.post(
                url,
                data=data,
                files=files,
                stream=True,
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"cannot reach the API at {url}. Is the service running?"
            ) from exc

    with response:
        if response.status_code != 200:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                detail = response.text[:200]
            if response.status_code == 409:
                raise RuntimeError(f"server busy (single concurrency): {detail}")
            raise RuntimeError(f"API error {response.status_code}: {detail}")

        sample_format = response.headers.get("X-Sample-Format", "s16le")
        if sample_format != "s16le":
            raise RuntimeError(f"unsupported sample format: {sample_format}")
        sample_rate = int(response.headers.get("X-Sample-Rate", FALLBACK_SAMPLE_RATE))

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_chunk = True
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for chunk in response.iter_content(chunk_size=16384):
                if first_chunk:
                    print(
                        f"time to first audio: {time.perf_counter() - started:.2f}s"
                        f" ({sample_rate} Hz s16le mono)",
                        flush=True,
                    )
                    first_chunk = False
                if chunk:
                    wav.writeframes(chunk)

    print(
        f"saved {output_path} ({read_wav_info(output_path).duration_s:.2f}s of audio)",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Breeze TTS 2 streaming API client")
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="natural-language voice description or direction",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="guidance scale; use 4 for stronger instruction following",
    )
    parser.add_argument("--ref-audio", type=Path, help="reference audio for cloning")
    parser.add_argument(
        "--ref-text", default="", help="exact transcript of the reference audio"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("speech.wav"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--wait", action="store_true", help="poll /health until the model has loaded"
    )
    args = parser.parse_args()

    if args.wait:
        wait_for_service(args.base_url)

    try:
        synthesize_speech(
            args.text,
            base_url=args.base_url,
            instruction=args.instruction,
            cfg_scale=args.cfg_scale,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            seed=args.seed,
            output=args.output,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
