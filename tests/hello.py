from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
from scipy.signal import resample
import io
import time
import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from openai import OpenAI

TIMEOUT = 1
DURATION = 5  # seconds
OUTPUT_FILE = "recorded_audio.wav"

# Initialize client
client = OpenAI(
    base_url="http://192.168.1.18:8000/v1",
    api_key="EMPTY"
)

def numpy_to_wav_bytes(audio_np, samplerate):
    """将 NumPy 音频数据转换为 WAV 格式的 bytes"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, samplerate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()

def transcribe_audio(audio_np, samplerate):
    """调用 OpenAI 兼容接口进行语音识别"""
    wav_bytes = numpy_to_wav_bytes(audio_np, samplerate)
    transcription = client.audio.transcriptions.create(
        model="./Qwen3-ASR-1.7B/",
        file=("audio.wav", wav_bytes, "audio/wav"),
    )
    return transcription.text

with ReachyMini() as mini: 
    print("Connected to Reachy Mini!")

    print("Wiggling antennas…") 
    # mini.goto_target(antennas=[0.5, -0.5], duration=0.5) 
    # mini.goto_target(antennas=[-0.5, 0.5], duration=0.5) 
    # mini.goto_target(antennas=[0, 0], duration=0.5)

    print("Done!")

    # Move everything at once
    # mini.goto_target(
    #     head=create_head_pose(z=10, mm=True),    # Up 10mm
    #     antennas=np.deg2rad([45, 45]),           # Antennas out
    #     body_yaw=np.deg2rad(30),                 # Turn body
    #     duration=2.0,                            # Take 2 seconds
    #     method="minjerk"                         # Smooth acceleration
    # )

    # frame = mini.media.get_frame()
    # print(frame.shape)
    audio_samples = []
    mini.media.start_recording()
    # Wait to actually get an audio sample
    print("Waiting for the microphone to be ready...")
    start_time = time.time()
    while (
        mini.media.get_audio_sample() is None and time.time() - start_time < TIMEOUT
    ):
        time.sleep(0.005)

    if time.time() - start_time >= TIMEOUT:
        print(f"Timeout: the microphone did not respond in {TIMEOUT} seconds.")
        # return

    print(f"Recording for {DURATION} seconds...")

    start_time = time.time()
    while time.time() - start_time < DURATION:
        sample = mini.media.get_audio_sample()
        if sample is not None:
            audio_samples.append(sample)

    mini.media.stop_recording()
    # print(sample)
    # Concatenate all samples and save
    if audio_samples:
        audio_data = np.concatenate(audio_samples, axis=0)
        # Convert stereo → mono if needed
        if audio_data.ndim == 2:
            mono = audio_data.mean(axis=1)
        else:
            mono = audio_data
        samplerate = mini.media.get_input_audio_samplerate()
        model = load_silero_vad()
        speech_timestamps = get_speech_timestamps(
            mono,
            model,
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
        print(f"检测到 {len(speech_timestamps)} 段语音: {speech_timestamps}")
        if speech_timestamps:
            # ===== 方案 A: 整段音频直接识别 =====
            text = transcribe_audio(audio_data, samplerate)
            print(f"识别结果: {text}")
            # ===== 方案 B: 只识别 VAD 检测到的语音段（更精准）=====
            # for i, ts in enumerate(speech_timestamps):
            #     start_sample = int(ts['start'] * samplerate)
            #     end_sample = int(ts['end'] * samplerate)
            #     speech_segment = audio_data[start_sample:end_sample]
            #     text = transcribe_audio(speech_segment, samplerate)
            #     print(f"第 {i+1} 段语音 [{ts['start']:.2f}s - {ts['end']:.2f}s]: {text}")
        sf.write(OUTPUT_FILE, audio_data, samplerate)
        print(f"Audio saved to {OUTPUT_FILE}")
    else:
        print("No audio data recorded.")