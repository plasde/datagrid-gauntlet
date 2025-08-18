import os
import azure.cognitiveservices.speech as speechsdk

def transcribe_file(path: str, key: str, region: str) -> str:
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    res = recognizer.recognize_once()
    if res.reason == speechsdk.ResultReason.RecognizedSpeech:
        return res.text
    raise RuntimeError(f"Transcription failed: {res.reason}")
