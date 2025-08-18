import os, json, time, sys
from pathlib import Path
from dotenv import load_dotenv

# Robust .env load from repo root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

# Local imports
sys.path.append(str(ROOT / "src"))
from utils import transcribe_file
from summarize_and_structurize import summarize_and_structurize

OUT = ROOT / "outputs"
AUDIO_DIR = ROOT / "data/subset"
OUT.mkdir(exist_ok=True, parents=True)

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "westeurope")

def main():
    audio_files = sorted([p for p in AUDIO_DIR.glob("*") if p.suffix.lower() in {".wav",".mp3",".flac"}])
    if not audio_files:
        raise SystemExit(f"No audio in {AUDIO_DIR}. Put your subset there.")

    results_path = OUT / "results.jsonl"
    n_ok = n_fail = 0

    with results_path.open("a", encoding="utf-8") as wf:
        for i, path in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {path.name}")
            try:
                transcript = transcribe_file(str(path), SPEECH_KEY, SPEECH_REGION)
                bullets, obj = summarize_and_structurize(transcript)

                record = {
                    "file": path.name,
                    "transcript": transcript,
                    "summary_bullets": bullets,
                    "structured": obj,
                }
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")

                # also save per-file JSON for convenience
                (OUT / f"{path.stem}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                n_ok += 1

                # gentle pacing for AOAI; tweak as needed
                time.sleep(0.3)
            except Exception as e:
                print(f"   ❌ failed: {e}")
                (OUT / f"{path.stem}.error.txt").write_text(str(e), encoding="utf-8")
                n_fail += 1

    print(f"\nDone: {n_ok} ok, {n_fail} failed. Aggregate: {results_path}")

if __name__ == "__main__":
    main()
