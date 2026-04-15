import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .models import QAAssessment

load_dotenv()


class QAAssessor:
    def __init__(self, config):
        self.config = config
        self.llm_config = config["llm_provider"]

        # Create base LLM with HTTP endpoint
        base_llm = ChatOpenAI(
            base_url=self.llm_config["base_url"],
            api_key=self.llm_config.get("api_key", "dummy"),
            model=self.llm_config["models"]["qa_assessor"],
            temperature=self.llm_config["parameters"]["temperature"],
            max_tokens=self.llm_config["parameters"]["max_tokens"],
        )

        # Add structured output - LangChain handles JSON parsing and validation
        self.llm = base_llm.with_structured_output(QAAssessment)

        # Simplified system prompt - no manual JSON formatting instructions needed
        self.system_prompt = """You are an expert quality assurance analyst for support call recordings.

Review the following support call transcript with labeled speakers (agent and customer) and provide a comprehensive QA assessment.

Rate each criterion on a 0-5 scale with brief reasoning:

1. Resolution Quality (0-5):
   - Did the agent successfully resolve the issue?
   - Was escalation needed?
   - Were clear next steps provided?
   - 5 = Excellent resolution, 0 = Failed to resolve

2. Tone/Phenomena (0-5):
   - Was the agent professional and empathetic?
   - Was there any negative sentiment or frustration?
   - Did the agent demonstrate good communication skills?
   - 5 = Excellent tone, 0 = Unprofessional or negative

3. Compliance (0-5):
   - Were there any data breaches or privacy violations?
   - Did the agent follow company policies and protocols?
   - Were GDPR/customer data protection guidelines followed?
   - 5 = Fully compliant, 0 = Major violations

4. Overall Rating (0-10):
   - Composite assessment of the entire call
   - Consider resolution quality, tone, and compliance
   - 10 = Excellent call, 0 = Poor call

Additionally provide:
- Call Reason: Brief description of why customer called
- Category: One of: Technical Support, Billing Inquiry, Account Setup, Feature Request, Complaint, Information Request, Other
- Summary: Concise summary of the call (2-3 sentences)
- Strengths: List 2-3 things the agent did well
- Improvements: List 2-3 areas where agent could improve"""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def assess_qa(self, labeled_transcript):
        print("Assessing QA...")

        context = self._prepare_context(labeled_transcript)

        try:
            chain = self.prompt | self.llm
            result = chain.invoke([("human", context)])

            # result is already a QAAssessment Pydantic model
            # Convert to clean dictionary for consistency
            output = self._pydantic_to_dict(result)

            print(
                f"QA assessment complete - Overall Rating: {output['overall_rating']['score']}/10"
            )
            return output

        except Exception as e:
            print(f"ERROR in QA assessment: {e}")
            raise RuntimeError(f"QA assessment failed: {e}")

    def _prepare_context(self, labeled_transcript):
        """Prepare transcript with role labels for LLM."""
        context = "Support call transcript with labeled speakers:\n\n"

        for segment in labeled_transcript:
            role = segment.get("role", "unknown")
            text = segment.get("text", "")
            context += f"[{role.upper()}] {text}\n"

        return context

    def _pydantic_to_dict(self, model: QAAssessment) -> dict:
        """Convert Pydantic model to clean dictionary."""
        return {
            "resolution_quality": {
                "score": model.resolution_quality.score,
                "reasoning": model.resolution_quality.reasoning,
            },
            "tone_phenomena": {
                "score": model.tone_phenomena.score,
                "reasoning": model.tone_phenomena.reasoning,
            },
            "compliance": {
                "score": model.compliance.score,
                "reasoning": model.compliance.reasoning,
            },
            "overall_rating": {
                "score": model.overall_rating.score,
                "reasoning": model.overall_rating.reasoning,
            },
            "call_reason": model.call_reason,
            "category": model.category,
            "summary": model.summary,
            "strengths": model.strengths,
            "improvements": model.improvements,
        }
