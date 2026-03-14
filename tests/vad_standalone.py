

import numpy as np
import soundfile as sf
import torch
# from openai import OpenAI
from silero_vad import load_silero_vad, get_speech_timestamps
# from reachy_mini import ReachyMini
# from reachy_mini.utils import create_head_pose

# ============ 配置参数 ============
SILENCE_THRESHOLD = 1.5    # 静音超过1.5秒认为说完了，送入ASR
MIN_SPEECH_DURATION = 0.3  # 最短语音段（秒），过短的丢弃（避免噪音误触发）
MAX_SPEECH_DURATION = 30   # 最长语音段（秒），超过强制送入ASR

SAMPLE_RATE = 16000


def is_speech(audio_chunk, model, sample_rate=16000):
    """
    判断一小段音频是否包含语音
    返回: 语音概率 (0.0 ~ 1.0)
    """
    audio_tensor = torch.from_numpy(audio_chunk).float()
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor[:, 0]  # 取第一个通道
    speech_prob = model(audio_tensor, sample_rate).item()
    return speech_prob


# ============ 状态机：核心逻辑 ============
class VADStateMachine:
    """
    状态机：
      WAITING  → 等待用户开口说话
      SPEAKING → 用户正在说话，持续收集音频
      SILENCE  → 检测到静音，等待是否继续说话
    """

    def __init__(self):
        self.state = "WAITING"
        self.speech_buffer = []        # 存储当前语音段的音频
        self.silence_start_time = None # 静音开始时间
        self.speech_start_time = None  # 语音开始时间
        self.speech_prob_threshold = 0.5  # 语音概率阈值
        self.vad_model = load_silero_vad()

    def process_chunk(self, audio_chunk, current_time):
        """
        处理一个音频块，返回:
          None          - 继续录音
          numpy_array   - 完整语音段，应送入ASR
        """
        prob = is_speech(audio_chunk, self.vad_model, SAMPLE_RATE)

        if self.state == "WAITING":
            if prob > self.speech_prob_threshold:
                # 检测到语音开始！
                print(f"🎤 检测到语音开始 (概率: {prob:.2f})")
                self.state = "SPEAKING"
                self.speech_start_time = current_time
                self.speech_buffer = [audio_chunk]
            return None

        elif self.state == "SPEAKING":
            self.speech_buffer.append(audio_chunk)

            if prob > self.speech_prob_threshold:
                # 仍在说话
                # 检查是否超过最大时长
                if current_time - self.speech_start_time > MAX_SPEECH_DURATION:
                    print(f"⚠️ 超过最大时长 {MAX_SPEECH_DURATION}s，强制送入ASR")
                    return self._flush()
            else:
                # 开始静音
                self.state = "SILENCE"
                self.silence_start_time = current_time

            return None

        elif self.state == "SILENCE":
            self.speech_buffer.append(audio_chunk)

            if prob > self.speech_prob_threshold:
                # 用户又开始说话了（只是短暂停顿）
                print(f"🎤 短暂停顿后继续说话")
                self.state = "SPEAKING"
                self.silence_start_time = None
                return None

            # 仍然是静音
            silence_duration = current_time - self.silence_start_time
            if silence_duration >= SILENCE_THRESHOLD:
                # 静音足够长，认为说完了！
                speech_duration = current_time - self.speech_start_time
                print(f"🔇 静音 {silence_duration:.1f}s，语音段时长: {speech_duration:.1f}s")

                if speech_duration - silence_duration < MIN_SPEECH_DURATION:
                    # 有效语音太短，丢弃
                    print(f"⚠️ 语音太短，丢弃")
                    self._reset()
                    return None

                return self._flush()

            return None

    def _flush(self):
        """输出完整语音段并重置状态"""
        audio_data = np.concatenate(self.speech_buffer, axis=0)
        self._reset()
        return audio_data

    def _reset(self):
        """重置状态机"""
        self.state = "WAITING"
        self.speech_buffer = []
        self.silence_start_time = None
        self.speech_start_time = None
        # 重置 VAD 模型的内部状态
        self.vad_model.reset_states()



