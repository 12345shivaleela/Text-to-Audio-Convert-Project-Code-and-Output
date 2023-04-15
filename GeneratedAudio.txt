from gtts import gTTS
from playsound import playsound
audio="generated audio.mp3"
language="te"
generateaudio=gTTS(text="తెలుగు అంట్టే చాలా ఎస్టాం",lang=language,slow=False)
generateaudio.save(audio)
playsound(audio)
print("======audio is playing======")
