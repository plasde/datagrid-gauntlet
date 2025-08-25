import os, json
from dotenv import load_dotenv
from json_schema import ClinicalNoteJSON
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

SYSTEM = """You are a clinical documentation assistant. 
Produce: 
1) a concise, bullet-point summary (<=5 bullets), 
2) a compact JSON object with fields: chief_complaint, diagnosis, plan, follow_up_instructions.
Be faithful to the transcript; don't invent facts. Keep wording clinician-ready."""

def summarize_and_structurize(transcript: str) -> tuple[str, dict]:
    user_prompt = f"""Transcript of dictated clinical note:
---
{transcript}
---
Return:
- Bulleted summary (<=5 bullets).
- Minimal JSON with keys: chief_complaint, diagnosis, plan, follow_up_instructions.
"""
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":user_prompt}],
        #temperature=0.2,
    )
    text = resp.choices[0].message.content

    # Heuristic split: bullets above, JSON below (fenced or inline). Robustify later.
    # Extract the last {...} block as JSON.
    import re
    match = list(re.finditer(r"\{(?:.|\n)*\}", text))
    if not match:
        raise ValueError("No JSON found in model output.")
    json_str = match[-1].group(0)
    data = json.loads(json_str)
    ClinicalNoteJSON(**data)  # validate
    bullets = text[:match[-1].start()].strip()
    return bullets, data

if __name__ == "__main__":
    sample_text = ("Chief complaint: headache x2 days with photophobia. "
                   "No neuro deficits. Assessment: migraine without aura. "
                   "Plan: ibuprofen PRN, hydration, dark room; return if neuro changes; "
                   "follow-up with PCP in 1 week.")
    bullets, obj = summarize_and_structurize(sample_text)
    print("BULLETS:\n", bullets)
    print("\nJSON:\n", json.dumps(obj, indent=2))
