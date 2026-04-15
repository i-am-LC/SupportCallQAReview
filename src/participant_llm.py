import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .models import ParticipantLabels

load_dotenv()


class ParticipantLabeler:
    def __init__(self, config):
        self.config = config
        self.llm_config = config["llm_provider"]

        # Create base LLM with HTTP endpoint
        base_llm = ChatOpenAI(
            base_url=self.llm_config["base_url"],
            api_key=self.llm_config.get("api_key", "dummy"),
            model=self.llm_config["models"]["participant_labeler"],
            temperature=self.llm_config["parameters"]["temperature"],
            max_tokens=self.llm_config["parameters"]["max_tokens"],
        )

        # Add structured output - LangChain handles JSON parsing and validation
        self.llm = base_llm.with_structured_output(ParticipantLabels)

        # Simplified system prompt - no manual JSON formatting instructions needed
        self.system_prompt = """You are an expert at identifying participants in support call recordings.

        Review the following full transcript and identify:
        1. Which speaker is the support agent and which is the customer
        2. The names of participants if they introduce themselves

        Support call characteristics:
        - The support agent typically introduces themselves first
        - The agent handles inquiries and provides assistance
        - The customer asks questions or describes issues
        - Support calls usually have clear role differentiation

        Name identification - look for phrases like:
        - "my name is [name]"
        - "this is [name]"
        - "I'm [name]"
        - "you can call me [name]"
        - "this is [name] from [company]"

        Output the speaker ID to role and name mapping."""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def label_participants(self, transcript, speaker_segments):
        print("Labeling participants...")

        context = self._prepare_context(transcript, speaker_segments)

        try:
            chain = self.prompt | self.llm
            result = chain.invoke([("human", context)])

            # result is already a ParticipantLabels Pydantic model
            # Normalize keys to match pyannote format (SPEAKER_0 -> SPEAKER_00, etc.)
            normalized_labels = {}
            for speaker_id, info in result.labels.items():
                if speaker_id.startswith("SPEAKER_"):
                    num = speaker_id.replace("SPEAKER_", "")
                    normalized_id = f"SPEAKER_{int(num):02d}"
                else:
                    normalized_id = speaker_id
                normalized_labels[normalized_id] = {
                    "role": info.role,
                    "name": info.name,
                }

            output = {"labels": normalized_labels}

            print(
                f"Participant labeling complete: {len(output['labels'])} speakers identified"
            )
            return output

        except Exception as e:
            print(f"ERROR in participant labeling: {e}")
            raise RuntimeError(f"Participant labeling failed: {e}")

    def _prepare_context(self, transcript, speaker_segments):
        """Prepare context for LLM from transcript and speaker segments."""
        context = f"Number of speaker segments: {len(speaker_segments)}\n"
        context += "Speaker segments identified:\n"

        for i, segment in enumerate(
            speaker_segments[:10]
        ):  # Limit to first 10 for efficiency
            context += f"  {segment['speaker_id']}: {segment['start']:.1f}s - {segment['end']:.1f}s\n"

        if len(speaker_segments) > 10:
            context += f"  ... and {len(speaker_segments) - 10} more segments\n"

        context += "\nFull transcript:\n"
        for i, segment in enumerate(transcript):
            speaker_id = segment.get("speaker_id", f"SPEAKER_{i % 2}")
            context += f"[{speaker_id}] {segment['text']}\n"

        return context
