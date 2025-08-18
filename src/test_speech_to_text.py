import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION", "westeurope")
audio_file = next(Path("data/subset").glob("*.*"), None)
if not audio_file:
    raise SystemExit("Put at least one audio file in data/subset first.")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
# Improve medical accuracy later with custom models; for now, baseline STT. :contentReference[oaicite:8]{index=8}
audio_config = speechsdk.audio.AudioConfig(filename=str(audio_file))
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

print(f"Transcribing: {audio_file.name}")
result = recognizer.recognize_once()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("TRANSCRIPT:\n", result.text)
elif result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech could be recognized.")
else:
    print("Recognition canceled:", result.cancellation_details.reason)
