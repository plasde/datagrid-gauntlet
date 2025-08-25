import os, sys, json, time, argparse, random
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root no matter where you run this
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
sys.path.append(str(ROOT / "src"))

# Local modules
from utils import transcribe_file

# Summarizer import is optional (only when --summarize is used)
def maybe_import_summarizer():
    from summarize_and_structurize import summarize_and_structurize  # noqa: F401
    return summarize_and_structurize

AUDIO_EXTS = {".wav", ".mp3", ".flac"}

def pick_files(base: Path, limit: int, shuffle: bool, seed: int):
    files = [p for p in base.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise SystemExit(f"No audio files found under {base.resolve()}")
    if shuffle:
        random.seed(seed); random.shuffle(files)
    return files[:limit]

def main():
    ap = argparse.ArgumentParser(description="Transcribe many audio files; optionally summarize to bullets+JSON.")
    ap.add_argument("--dir", default=str(ROOT / "data/subset"), help="Folder containing audio files")
    ap.add_argument("--limit", "-n", type=int, default=10, help="How many files to process")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before picking files")
    ap.add_argument("--seed", type=int, default=42, help="Seed for shuffling")
    ap.add_argument("--summarize", action="store_true", help="Also call Azure OpenAI to summarize + structure")
    ap.add_argument("--out", default=str(ROOT / "outputs/test_many.jsonl"), help="JSONL output when --summarize is set")
    ap.add_argument("--sleep", type=float, default=0.2, help="Pause between requests (helps avoid rate limits)")
    args = ap.parse_args()

    audio_dir = Path(args.dir)
    files = pick_files(audio_dir, args.limit, args.shuffle, args.seed)
    print(f"Found {len(files)} file(s) to process from {audio_dir.resolve()}")

    # Speech config (from .env)
    SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "westeurope")
    if not SPEECH_KEY:
        raise SystemExit("Missing AZURE_SPEECH_KEY in .env")

    # Optional summarizer
    summarize_and_structurize = None
    if args.summarize:
        # Small print to confirm which AOAI endpoint/deployment you'll use
        print("Summarizer enabled.")
        print("  AOAI endpoint  :", os.getenv("AZURE_OPENAI_ENDPOINT"))
        print("  AOAI deployment:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        print("  AOAI API ver   :", os.getenv("AZURE_OPENAI_API_VERSION"))
        summarize_and_structurize = maybe_import_summarizer()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fout = open(args.out, "a", encoding="utf-8")
    else:
        fout = None

    ok = fail = 0
    for i, path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {path.name}")
        try:
            # 1) Transcribe
            transcript = transcribe_file(str(path), SPEECH_KEY, SPEECH_REGION)
            print("TRANSCRIPT:", transcript)

            # 2) Summarize + structure (optional)
            if summarize_and_structurize:
                try:
                    bullets, obj = summarize_and_structurize(transcript)
                    print("SUMMARY (≤5 bullets):\n", bullets)
                    print("JSON:\n", json.dumps(obj, ensure_ascii=False))
                    # write JSONL record
                    record = {
                        "file": path.name,
                        "transcript": transcript,
                        "summary_bullets": bullets,
                        "structured": obj,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print("  ⚠️ summarizer failed:", e)

            ok += 1
            time.sleep(args.sleep)
        except Exception as e:
            print("  ❌ failed:", e)
            fail += 1

    if fout: fout.close()
    print(f"\nDone. {ok} ok, {fail} failed.")
    if fout:
        print(f"JSONL written to: {args.out}")

if __name__ == "__main__":
    main()
