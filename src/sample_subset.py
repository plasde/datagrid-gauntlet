import os, random, shutil, sys
from pathlib import Path

RAW = Path("data/raw")            # point this at your Kaggle download root
SUBSET = Path("data/subset")
SUBSET.mkdir(parents=True, exist_ok=True)

# Collect audio files (commonly .wav in this dataset)
candidates = [p for p in RAW.rglob("*") if p.suffix.lower() in {".wav", ".mp3", ".flac"}]
if not candidates:
    sys.exit("No audio files found under data/raw. Put Kaggle dataset there.")

random.seed(42)
k = min(100, max(50, len(candidates)//50))  # target 50–100; adjusts if dataset smaller
subset = random.sample(candidates, k)

for src in subset:
    dst = SUBSET / src.name
    shutil.copy2(src, dst)

print(f"Copied {len(subset)} files to {SUBSET}")
