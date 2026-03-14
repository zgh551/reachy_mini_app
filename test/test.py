from silero_vad import load_silero_vad, read_audio

wav = read_audio("recorded_audio.wav", sampling_rate=16000)

print(wav)