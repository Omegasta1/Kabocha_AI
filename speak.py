# core/speak.py
from chatterbox.tts import ChatterboxTTS
import soundfile as sf
import os
import sounddevice as sd
import soundfile as sf

class KaboTTS:
    def __init__(self, device: str = "cuda", audio_prompt_path: str = None):
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.audio_prompt_path = audio_prompt_path

    def speak(self, text: str, output_path: str = None):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), "output.wav")
        if self.audio_prompt_path:
            wav = self.model.generate(text, audio_prompt_path=self.audio_prompt_path)
        else:
            wav = self.model.generate(text)
        sf.write(output_path, wav[0], self.model.sr)
        print(f"Audio gespeichert unter: {output_path}")

def play_audio(file_path: str):
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()

# Beispielverwendung
if __name__ == "__main__":
    # Pfad zur Referenz-Audiodatei im Verzeichnis "models"
    reference_audio = os.path.join(os.path.dirname(__file__), "..", "models", "Kikuri_VA_Sample.wav")
    tts = KaboTTS(audio_prompt_path=reference_audio)
    tts.speak("Hello there!")
