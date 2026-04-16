import os
import json
import random
from datetime import datetime
from pathlib import Path


def load_output_files(output_dir):
    """Load all JSON files from output directory."""
    calls = []
    output_path = Path(output_dir)

    if not output_path.exists():
        return calls

    for json_file in output_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["_source_file"] = json_file.name
                calls.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping {json_file.name}: {e}")

    return calls


def group_by_agent(calls):
    """Group calls by agent name."""
    agents = {}

    for call in calls:
        participants = call.get("participants", [])

        agent_name = None
        for participant in participants:
            role = participant.get("role", "").lower()
            if role in ["agent", "support_agent"]:
                agent_name = participant.get("name")
                break

        if not agent_name:
            agent_name = "Unknown"

        if agent_name not in agents:
            agents[agent_name] = []

        agents[agent_name].append(call)

    return agents


def get_random_calls(calls, max_calls=4):
    """Select up to max_calls randomly from available calls."""
    if len(calls) <= max_calls:
        return calls
    return random.sample(calls, max_calls)


def format_transcript(call):
    """Format transcript as conversation with role tags."""
    full_transcript = call.get("full_transcript", [])
    participants = {
        p["speaker_id"]: p.get("name", p.get("role", "Unknown"))
        for p in call.get("participants", [])
    }

    lines = []
    for segment in full_transcript:
        speaker_id = segment.get("speaker_id", "UNKNOWN")
        name = participants.get(speaker_id, speaker_id)
        text = segment.get("text", "")
        lines.append(f"[{name}] {text}")

    return "\n".join(lines) if lines else "No transcript available"


def format_qa_assessment(qa):
    """Format QA assessment section."""
    lines = []

    resolution = qa.get("resolution_quality", {})
    lines.append(
        f"- **Resolution Quality**: {resolution.get('score', 'N/A')}/5 - {resolution.get('reasoning', '')}"
    )

    tone = qa.get("tone_phenomena", {})
    lines.append(
        f"- **Tone/Phenomena**: {tone.get('score', 'N/A')}/5 - {tone.get('reasoning', '')}"
    )

    compliance = qa.get("compliance", {})
    lines.append(
        f"- **Compliance**: {compliance.get('score', 'N/A')}/5 - {compliance.get('reasoning', '')}"
    )

    overall = qa.get("overall_rating", {})
    lines.append(
        f"- **Overall Rating**: {overall.get('score', 'N/A')}/10 - {overall.get('reasoning', '')}"
    )

    return "\n".join(lines)


def format_call_analysis(call):
    """Format call analysis section."""
    analysis = call.get("call_analysis", {})
    lines = []

    reason = analysis.get("reason")
    if reason:
        lines.append(f"- **Reason**: {reason}")

    category = analysis.get("category")
    if category:
        lines.append(f"- **Category**: {category}")

    summary = analysis.get("summary")
    if summary:
        lines.append(f"- **Summary**: {summary}")

    strengths = analysis.get("strengths", [])
    if strengths:
        lines.append(f"- **Strengths**: {', '.join(strengths)}")

    improvements = analysis.get("improvements", [])
    if improvements:
        lines.append(f"- **Improvements**: {', '.join(improvements)}")

    return "\n".join(lines)


def format_agent_report(agent_name, calls):
    """Format report for a single agent."""
    if not calls:
        return ""

    selected_calls = get_random_calls(calls, max_calls=3)

    lines = [f"\n## {agent_name} ({len(selected_calls)} calls analyzed)\n"]

    for i, call in enumerate(selected_calls, 1):
        metadata = call.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        qa = call.get("qa_assessment", {})
        overall = qa.get("overall_rating", {})
        score = overall.get("score", "N/A")
        category = call.get("call_analysis", {}).get("category", "Unknown")

        participants = call.get("participants", [])
        participant_names = [
            p.get("name") or p.get("role", "Unknown") for p in participants
        ]
        participants_str = ", ".join(str(p) for p in participant_names if p)

        lines.append(f"### Call #{i}")
        lines.append(f"**Call**: {filename}")
        lines.append(f"**Score**: {score}/10")
        lines.append(f"**Category**: {category}")
        lines.append(f"**Participants**: {participants_str}")
        lines.append("")
        lines.append("#### Transcript")
        lines.append(format_transcript(call))
        lines.append("")
        lines.append("#### QA Assessment")
        lines.append(format_qa_assessment(qa))
        lines.append("")
        lines.append("#### Call Analysis")
        lines.append(format_call_analysis(call))

    return "\n".join(lines)


def generate_report(output_dir):
    """Generate complete markdown report for all agents."""
    calls = load_output_files(output_dir)

    if not calls:
        print("No output files found to generate report.")
        return None

    agents = group_by_agent(calls)

    if not agents:
        print("No agent data found in output files.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        "# Support Rep Performance Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Calls Analyzed: {len(calls)}",
        "",
        "---",
        "",
    ]

    for agent_name in sorted(agents.keys()):
        agent_calls = agents[agent_name]
        report_lines.append(format_agent_report(agent_name, agent_calls))

    report_content = "\n".join(report_lines)

    report_filename = f"support_rep_report_{timestamp}.md"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/"
    generate_report(output_dir)
