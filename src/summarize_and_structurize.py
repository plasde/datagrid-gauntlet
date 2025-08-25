import os, json, re
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ValidationError
from json_schema import ClinicalNoteJSON
from openai import AzureOpenAI

# --- env ---
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
endpoint  = os.getenv("AZURE_OPENAI_ENDPOINT","").strip()
api_key   = os.getenv("AZURE_OPENAI_API_KEY","").strip()
api_ver   = os.getenv("AZURE_OPENAI_API_VERSION","2025-01-01-preview").strip()
DEPLOYMENT= os.getenv("AZURE_OPENAI_DEPLOYMENT","").strip()

def _require(n, v):
    if not v: raise SystemExit(f"Missing {n} in .env")
_require("AZURE_OPENAI_ENDPOINT", endpoint)
_require("AZURE_OPENAI_API_KEY", api_key)
_require("AZURE_OPENAI_DEPLOYMENT", DEPLOYMENT)
if not re.search(r"\.openai\.azure\.com/?$", endpoint):
    raise SystemExit(f"Endpoint looks wrong: {endpoint}")

client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_ver)

SYSTEM = (
    "You are a clinical documentation assistant. Be faithful to the transcript; "
    "do NOT invent facts. Always populate every field. If info is not present, "
    "use the string 'unknown' (not empty). Keep phrasing concise."
)

# Function/tool schema enforces required fields and non-empty strings
TOOL = {
  "type": "function",
  "function": {
    "name": "emit_clinical_json",
    "description": "Return structured clinical info derived from the transcript.",
    "parameters": {
      "type": "object",
      "properties": {
        "summary_bullets": {"type":"string","minLength":1, "description":"<=5 concise bullets."},
        "chief_complaint": {"type":"string","minLength":1},
        "diagnosis": {"type":"string","minLength":1},
        "plan": {"type":"string","minLength":1},
        "follow_up_instructions": {"type":"string","minLength":1}
      },
      "required": ["summary_bullets","chief_complaint","diagnosis","plan","follow_up_instructions"],
      "additionalProperties": False
    }
  }
}

def _fill_unknowns(d: dict) -> dict:
    for k in ("chief_complaint","diagnosis","plan","follow_up_instructions","summary_bullets"):
        if not str(d.get(k, "")).strip():
            d[k] = "unknown"
    return d

def summarize_and_structurize(transcript: str):
    user = f"""Transcript of dictated clinical note:
---
{transcript}
---
Rules:
- Always fill every field; if missing in the transcript, write 'unknown'.
- Keep 'summary_bullets' to 5 bullets max; short phrases, no extra commentary.
"""
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":user}],
        tools=[TOOL],
        tool_choice={"type":"function","function":{"name":"emit_clinical_json"}},
        # NOTE: no temperature here; o4-mini preview doesn't allow it
    )

    tcalls = resp.choices[0].message.tool_calls or []
    if not tcalls:
        raise ValueError("Model did not return a function call with JSON.")

    args_json = tcalls[0].function.arguments
    data = json.loads(args_json)
    data = _fill_unknowns(data)

    try:
        ClinicalNoteJSON(
            chief_complaint=data["chief_complaint"],
            diagnosis=data["diagnosis"],
            plan=data["plan"],
            follow_up_instructions=data["follow_up_instructions"]
        )
    except ValidationError as e:
        # As a last resort, coerce empties to 'unknown' and continue
        data = _fill_unknowns(data)

    bullets = data.pop("summary_bullets", "unknown")
    return bullets, data

if __name__ == "__main__":
    sample_text = ("Chief complaint: headache x2 days with photophobia. "
                   "Assessment: likely migraine without aura. "
                   "Plan: ibuprofen PRN, hydration, dark room; "
                   "follow-up with PCP in 1 week; return if neuro changes.")
    b, j = summarize_and_structurize(sample_text)
    print("BULLETS:\n", b)
    print("\nJSON:\n", json.dumps(j, indent=2, ensure_ascii=False))
