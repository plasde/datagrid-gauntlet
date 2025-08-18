# Problem
Clinicians dictate free-form notes. We need to turn speech into concise, structured outputs that are instantly actionable.

# Goal (5 days)
An Azure-hosted app that:

1. accepts doctor audio,

2. transcribes (Azure Speech),

3. summarizes to bullets,

4. returns JSON: chief_complaint, diagnosis, plan, follow_up_instructions.

# Dataset
Kaggle: Medical Speech, Transcription, and Intent (thousands of audio utterances; ≈8.5 hours). Link in code comments.

# Day-1 Deliverables

- 50–100 audio files sampled to data/subset/

- Azure Speech + Azure OpenAI resources provisioned

- test_speech_to_text.py runs and prints a transcript

# Security & PHI
This dataset is synthetic/public; for real PHI later: store keys in Key Vault, avoid logging raw PHI, and restrict network egress.
