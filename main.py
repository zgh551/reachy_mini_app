from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time
import io
import queue

import numpy as np
import numpy.typing as npt

from .vad import VADStateMachine
from .asr import Qwen3ASR
from .llm import LLMClient
from .tts import Qwen3TTS

SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512       # VAD 每次处理的样本数（Silero要求512或1024，16kHz下）
TIMEOUT = 10
SENTENCE_ENDS = set("。！？.!?")

def _find_sentence_end(text: str) -> int:
    """Return the index of the first sentence-ending punctuation, or -1."""
    for i, ch in enumerate(text):
        if ch in SENTENCE_ENDS:
            return i
    return -1

def _push_audio(
    media,
    audio: npt.NDArray[np.float32],
    chunk_size: int = 1600,
) -> None:
    """Push audio to the speaker in small chunks.

    The robot speaker expects float32 stereo at 16 kHz.  We duplicate the
    mono channel to produce stereo.
    """
    # # Duplicate mono → stereo
    # stereo = np.stack([audio, audio], axis=-1)  # shape: (N, 2)
    media.push_audio_sample(audio)
    # offset = 0
    # while offset < len(stereo):
    #     chunk = stereo[offset : offset + chunk_size]
    #     media.push_audio_sample(chunk)
    #     offset += chunk_size


# ============ 主循环 ============
def main():
    vad = VADStateMachine()
    asr = Qwen3ASR()
    llm = LLMClient("http://192.168.200.252:30000/v1", "no-needed", "qwen3.5-27b")
    qwen3_tts = Qwen3TTS()
    motion_queue: queue.Queue[dict] = queue.Queue()

    with ReachyMini() as mini: 
        mini.media.start_recording()

        # 等待麦克风就绪
        print("等待麦克风就绪...")
        start_time = time.time()
        while mini.media.get_audio_sample() is None and time.time() - start_time < TIMEOUT:
            time.sleep(0.005)

        samplerate = mini.media.get_input_audio_samplerate()
        print(f"麦克风就绪，采样率: {samplerate}Hz")
        print("开始监听，请说话... (Ctrl+C 退出)")

        # 如果设备采样率不是16kHz，需要重采样
        need_resample = (samplerate != SAMPLE_RATE)
        if need_resample:
            import librosa
            print(f"注意: 设备采样率 {samplerate}Hz，将重采样到 {SAMPLE_RATE}Hz")

        audio_accumulator = np.array([], dtype=np.float32)

        try:
            while True:
                sample = mini.media.get_audio_sample()
                if sample is None:
                    time.sleep(0.005)
                    continue

                # 确保是一维 float32
                # chunk = sample.astype(np.float32).flatten()
                        # Convert stereo → mono if needed
                if sample.ndim == 2:
                    mono = sample.mean(axis=1).astype(np.float32)
                else:
                    mono = sample.astype(np.float32)

                # 重采样（如果需要）
                # if need_resample:
                #     import librosa
                #     chunk = librosa.resample(chunk, orig_sr=samplerate, target_sr=SAMPLE_RATE)

                # 累积到缓冲区
                audio_accumulator = np.concatenate([audio_accumulator, mono])

                # 每凑够 VAD_CHUNK_SIZE 个样本就处理一次
                while len(audio_accumulator) >= VAD_CHUNK_SIZE:
                    vad_chunk = audio_accumulator[:VAD_CHUNK_SIZE]
                    audio_accumulator = audio_accumulator[VAD_CHUNK_SIZE:]

                    current_time = time.time()
                    result = vad.process_chunk(vad_chunk, current_time)

                    if result is not None:
                        # 收到完整语音段，送入 ASR！
                        print("📡 正在识别...")
                        text = asr.transcribe_audio(result, SAMPLE_RATE)
                        print(f"✅ 识别结果: {text}")
                        print("-" * 50)
                        print("继续监听...")
                        if "小白" in text:
                            sentence_buf = ""
                            mini.media.start_playing()
                            try:
                                for token in llm.stream_response(text, motion_queue):
                                    sentence_buf += token
                                    
                                    # Flush complete sentences to TTS
                                    while True:
                                        end_idx = _find_sentence_end(sentence_buf)
                                        if end_idx == -1:
                                            break
                                        sentence = sentence_buf[: end_idx + 1].strip()
                                        sentence_buf = sentence_buf[end_idx + 1 :]

                                        if sentence:
                                            print("TTS: %s", sentence)
                                            audio_out = qwen3_tts.synthesize(sentence)
                                            if len(audio_out) > 0:
                                                # Push in chunks for smooth streaming
                                                _push_audio(mini.media, audio_out)
                                                time.sleep(len(audio_out) / mini.media.get_output_audio_samplerate())
                                print(sentence_buf)
                                # Flush any remaining text
                                if sentence_buf.strip():
                                    audio_out = qwen3_tts.synthesize(sentence_buf.strip())
                                    if len(audio_out) > 0:
                                        _push_audio(mini.media, audio_out)
                                        time.sleep(len(audio_out) / mini.media.get_output_audio_samplerate())

                            finally:
                                # Small buffer to let speaker drain before stopping
                                time.sleep(0.5)
                                mini.media.stop_playing()



        except KeyboardInterrupt:
            print("\n停止监听")
        finally:
            mini.media.stop_recording()


if __name__ == "__main__":
    main()