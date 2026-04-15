from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class ParticipantInfo(BaseModel):
    role: str = Field(description="'agent' or 'customer'")
    name: Optional[str] = Field(description="Name if provided, else null")


class ParticipantLabels(BaseModel):
    labels: Dict[str, ParticipantInfo] = Field(
        description="Mapping of speaker IDs to participant info (role and name)"
    )


class CriterionScore(BaseModel):
    score: int = Field(ge=0, le=5, description="Score from 0 to 5")
    reasoning: str = Field(description="Explanation for the score")


class OverallRating(BaseModel):
    score: int = Field(ge=0, le=10, description="Overall rating from 0 to 10")
    reasoning: str = Field(description="Explanation for the rating")


class QAAssessment(BaseModel):
    resolution_quality: CriterionScore = Field(
        description="Assessment of issue resolution quality"
    )
    tone_phenomena: CriterionScore = Field(
        description="Assessment of agent tone and phenomena"
    )
    compliance: CriterionScore = Field(
        description="Assessment of compliance and policy adherence"
    )
    overall_rating: OverallRating = Field(description="Overall composite rating")
    call_reason: str = Field(description="Brief description of why customer called")
    category: str = Field(
        description="Call category",
        examples=[
            "Technical Support",
            "Billing Inquiry",
            "Account Setup",
            "Feature Request",
            "Complaint",
            "Information Request",
            "Other",
        ],
    )
    summary: str = Field(description="2-3 sentence summary of the call")
    strengths: List[str] = Field(description="2-3 things the agent did well")
    improvements: List[str] = Field(description="2-3 areas for improvement")
