from pydantic import BaseModel, Field, field_validator

class ClinicalNoteJSON(BaseModel):
    chief_complaint: str = Field(..., description="Primary symptom or reason for visit.")
    diagnosis: str = Field(..., description="Most likely diagnosis or assessment.")
    plan: str = Field(..., description="Planned interventions, meds, tests, referrals.")
    follow_up_instructions: str = Field(..., description="Timing and guidance for follow-up.")

    @field_validator("*", mode="before")
    @classmethod
    def _non_empty(cls, v):
        v = (v or "").strip()
        return v if v else "unknown"

