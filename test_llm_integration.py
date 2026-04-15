import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from participant_llm import ParticipantLabeler
from qa_llm import QAAssessor


def test_participant_labeler():
    print("Testing ParticipantLabeler...")

    # Mock config
    config = {
        "llm_provider": {
            "base_url": "http://localhost:8080/v1",
            "api_key": "dummy",
            "models": {"participant_labeler": "gemma-2-4b-q4"},
            "parameters": {"temperature": 0.0, "max_tokens": 2048},
        }
    }

    # Mock transcript
    mock_transcript = [
        {
            "start": 0,
            "end": 5,
            "text": "Hello, thank you for calling support, my name is Sarah.",
        },
        {"start": 5, "end": 8, "text": "Hi, I'm having trouble with my account."},
        {"start": 8, "end": 12, "text": "I understand, let me help you with that."},
        {"start": 12, "end": 15, "text": "Thanks, I appreciate your help."},
    ]

    # Mock speaker segments
    mock_speaker_segments = [
        {"speaker_id": "SPEAKER_00", "start": 0, "end": 5},
        {"speaker_id": "SPEAKER_01", "start": 5, "end": 8},
        {"speaker_id": "SPEAKER_00", "start": 8, "end": 12},
        {"speaker_id": "SPEAKER_01", "start": 12, "end": 15},
    ]

    try:
        labeler = ParticipantLabeler(config)
        labels = labeler.label_participants(mock_transcript, mock_speaker_segments)

        # Labels should be a dictionary
        assert isinstance(labels, dict), "Labels should be a dictionary"
        assert "labels" in labels, "Labels should contain 'labels' key"

        # Each label should have a role of "agent" or "customer"
        for speaker_id, info in labels["labels"].items():
            assert info["role"] in ["agent", "customer"], (
                f"Invalid role: {info['role']} for {speaker_id}"
            )
        # Name should be string or None
        for speaker_id, info in labels["labels"].items():
            assert info["name"] is None or isinstance(info["name"], str), (
                f"Invalid name: {info['name']} for {speaker_id}"
            )

        print("✓ ParticipantLabeler test passed")
        print(f"  Labels: {labels['labels']}")
        return True
    except Exception as e:
        print(f"✗ ParticipantLabeler test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_qa_assessor():
    print("Testing QAAssessor...")

    # Mock config
    config = {
        "llm_provider": {
            "base_url": "http://localhost:8080/v1",
            "api_key": "dummy",
            "models": {"qa_assessor": "qwen2.5-4b-q4"},
            "parameters": {"temperature": 0.0, "max_tokens": 2048},
        }
    }

    # Mock labeled transcript
    mock_labeled_transcript = [
        {
            "role": "agent",
            "text": "Hello, thank you for calling support, my name is Sarah.",
        },
        {"role": "customer", "text": "Hi, I'm having trouble with my account."},
        {"role": "agent", "text": "I understand, let me help you with that issue."},
        {"role": "customer", "text": "Great, thank you for your help!"},
    ]

    try:
        assessor = QAAssessor(config)
        assessment = assessor.assess_qa(mock_labeled_transcript)

        # Assessment should be a dictionary
        assert isinstance(assessment, dict), "Assessment should be a dictionary"

        # Validate all required fields
        required_fields = [
            "resolution_quality",
            "tone_phenomena",
            "compliance",
            "overall_rating",
            "call_reason",
            "category",
            "summary",
            "strengths",
            "improvements",
        ]

        for field in required_fields:
            assert field in assessment, f"Assessment missing required field: {field}"

        # Validate score fields have range 0-5
        for score_field in ["resolution_quality", "tone_phenomena", "compliance"]:
            assert "score" in assessment[score_field], f"{score_field} missing score"
            assert "reasoning" in assessment[score_field], (
                f"{score_field} missing reasoning"
            )
            score = assessment[score_field]["score"]
            assert 0 <= score <= 5, f"{score_field} score {score} out of range 0-5"

        # Validate overall rating has range 0-10
        assert "score" in assessment["overall_rating"], "overall_rating missing score"
        assert "reasoning" in assessment["overall_rating"], (
            "overall_rating missing reasoning"
        )
        overall_score = assessment["overall_rating"]["score"]
        assert 0 <= overall_score <= 10, (
            f"Overall rating score {overall_score} out of range 0-10"
        )

        # Validate list fields
        assert isinstance(assessment["strengths"], list), "strengths should be a list"
        assert isinstance(assessment["improvements"], list), (
            "improvements should be a list"
        )
        assert len(assessment["strengths"]) >= 0, "strengths should not be empty"
        assert len(assessment["improvements"]) >= 0, "improvements should not be empty"

        # Validate string fields
        assert isinstance(assessment["call_reason"], str), (
            "call_reason should be a string"
        )
        assert isinstance(assessment["category"], str), "category should be a string"
        assert isinstance(assessment["summary"], str), "summary should be a string"

        print("✓ QAAssessor test passed")
        print(f"  Overall Rating: {assessment['overall_rating']['score']}/10")
        print(f"  Category: {assessment['category']}")
        print(f"  Resolution Quality: {assessment['resolution_quality']['score']}/5")
        print(f"  Tone/Phenomena: {assessment['tone_phenomena']['score']}/5")
        print(f"  Compliance: {assessment['compliance']['score']}/5")
        return True
    except Exception as e:
        print(f"✗ QAAssessor test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Phase 3: HTTP LLM Integration Testing")
    print("                    using LangChain Structured Output")
    print("=" * 60)
    print()

    # Note: These tests require a running HTTP LLM endpoint
    print("NOTE: These tests require a running LLM HTTP endpoint.")
    print("If tests fail, ensure your endpoint is running and accessible.")
    print()
    print("Expected endpoints:")
    print("  - llama.cpp: http://localhost:8080/v1")
    print("  - vLLM: http://localhost:8000/v1")
    print("  - Ollama: http://localhost:11434/v1")
    print()

    results = []

    # Test ParticipantLabeler
    try:
        result = test_participant_labeler()
        results.append(("ParticipantLabeler", result))
    except Exception as e:
        print(f"✗ ParticipantLabeler test errored: {e}")
        import traceback

        traceback.print_exc()
        results.append(("ParticipantLabeler", False))

    print()

    # Test QAAssessor
    try:
        result = test_qa_assessor()
        results.append(("QAAssessor", result))
    except Exception as e:
        print(f"✗ QAAssessor test errored: {e}")
        import traceback

        traceback.print_exc()
        results.append(("QAAssessor", False))

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)

    print()
    print(f"Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("✓ All tests completed successfully!")
        return 0
    else:
        print("✗ Some tests failed - check your HTTP endpoint configuration")
        return 1


if __name__ == "__main__":
    sys.exit(main())
