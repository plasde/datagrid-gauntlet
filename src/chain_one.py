import os, sys, json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
sys.path.append(str(ROOT / "src"))

from utils import transcribe_file
from summarize_and_structurize import summarize_and_structurize

SPEECH_KEY   = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION= os.getenv("AZURE_SPEECH_REGION", "westeurope")

def pick_audio():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]); 
        if not p.exists(): raise SystemExit(f"No such file: {p}")
        return p
    # fallback: first file in subset
    for ext in (".wav",".mp3",".flac"):
        hit = next((ROOT/"data/subset").glob(f"*{ext}"), None)
        if hit: return hit
    raise SystemExit("No audio found. Pass a path or put files in data/subset.")

audio = pick_audio()
print(f"Transcribing: {audio.name}")
tx = transcribe_file(str(audio), SPEECH_KEY, SPEECH_REGION)

print("\nTRANSCRIPT:\n", tx)
bullets, obj = summarize_and_structurize(tx)

print("\nSUMMARY (≤5 bullets):\n", bullets)
print("\nJSON:\n", json.dumps(obj, indent=2, ensure_ascii=False))

