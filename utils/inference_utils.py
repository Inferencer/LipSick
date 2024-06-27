import subprocess
import math

from pydub import AudioSegment
from pydub.silence import detect_silence

def boost_audio(audio):
    filter = "volume=1.5"
    boosted_audio_for_silence = audio.replace(".wav", "boosted.wav")
    cmd = f"ffmpeg -loglevel error -y -i {audio} -af {filter} {boosted_audio_for_silence}"
    subprocess.call(cmd, shell=True)
    return boosted_audio_for_silence

def detect_silence_pydub(audio):
    audio_segment = AudioSegment.from_wav(audio)
    silence_segments = detect_silence(
        audio_segment, min_silence_len=500, silence_thresh=-40)
    array = []
    for i, (start, end) in enumerate(silence_segments):
        array.append({"start":  math.ceil(start / 1000.0),
                     "end": round(end / 1000.0, 2)})
    return array
