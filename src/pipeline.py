import os
import json
import time
from datetime import datetime
from pathlib import Path
import logging

from .audio_processor import AudioProcessor
from .participant_llm import ParticipantLabeler
from .qa_llm import QAAssessor
from .models import ParticipantLabels, QAAssessment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.output_directory = config["directories"]["output"]

        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

        # Initialize components
        logger.info("Initializing pipeline components...")
        self.audio_processor = AudioProcessor(config)
        self.participant_llm = ParticipantLabeler(config)
        self.qa_llm = QAAssessor(config)
        logger.info("Pipeline initialized successfully")

    def _format_stage_header(self, stage_num, total, stage_name, status, duration):
        """Format stage header with Unicode box and timing."""
        status_icon = (
            "✅" if status == "success" else "❌" if status == "failed" else "⏳"
        )
        duration_str = f"({duration:.1f}s)"
        # Calculate padding for centered stage name
        name_spacing = 37 - len(stage_name) - len(f"Stage {stage_num}/{total}: ")
        spacing = " " * max(name_spacing, 0)
        line1 = f"┌─ Stage {stage_num}/{total}: {stage_name}{spacing}─┐ {status_icon} {duration_str}"
        return line1

    def _format_stage_body(self, bullet_points):
        """Format stage body with bullet points."""
        lines = []
        for point in bullet_points:
            # Pad line to match box width
            padded = point.ljust(43)
            lines.append(f"│  {padded}│")
        return lines

    def _format_stage_footer(self):
        """Format stage footer."""
        return "└───────────────────────────────────────────────────┘"

    def _display_stage_completion(
        self, stage_num, total, stage_name, duration, details
    ):
        """Display successfully completed stage with visual formatting."""
        logger.info(
            self._format_stage_header(stage_num, total, stage_name, "success", duration)
        )
        for line in self._format_stage_body(details):
            logger.info(line)
        logger.info(self._format_stage_footer())

    def _display_stage_failure(
        self, stage_num, total, stage_name, duration, error_msg, context_details
    ):
        """Display failed stage with visual formatting and error context."""
        logger.info(
            self._format_stage_header(stage_num, total, stage_name, "failed", duration)
        )

        # Add error details
        details = [f"Error: {error_msg}"]
        details.extend([f"  • {detail}" for detail in context_details])

        for line in self._format_stage_body(details):
            logger.info(line)
        logger.info(self._format_stage_footer())

    def process_file(self, audio_path):
        """
        Process a single audio file through the complete pipeline.

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: Complete JSON output
        """
        start_time = datetime.now()
        filename = Path(audio_path).name
        stage_timings = {}
        stage_details = {}

        logger.info(f"Processing file: {filename}")

        try:
            # Step 1: Audio processing (transcription + diarization)
            stage_start = time.time()
            audio_result = self.audio_processor.process(audio_path)
            transcript = audio_result["transcript"]
            speaker_segments = audio_result["speaker_segments"]
            stage1_time = time.time() - stage_start
            stage_timings["stage1"] = stage1_time

            if not transcript:
                raise ValueError("Transcription failed - no segments generated")

            if not speaker_segments:
                speaker_segments = self._create_speaker_segments_from_transcript(
                    transcript
                )

            stage1_details = [
                f"Generated {len(transcript)} transcript segments",
                f"Identified {len(speaker_segments)} speaker segments",
            ]
            stage_details["stage1"] = {"status": "success", "details": stage1_details}

            self._display_stage_completion(
                1, 5, "Audio processing", stage1_time, stage1_details
            )

            # Step 2: Participant labeling
            stage_start = time.time()
            labels_result = self.participant_llm.label_participants(
                transcript, speaker_segments
            )
            speaker_labels = labels_result["labels"]
            stage2_time = time.time() - stage_start
            stage_timings["stage2"] = stage2_time

            stage2_details = [f"Labeled {len(speaker_labels)} speakers"]
            stage_details["stage2"] = {"status": "success", "details": stage2_details}

            self._display_stage_completion(
                2, 5, "Participant labeling", stage2_time, stage2_details
            )

            # Step 3: Apply labels to transcript
            stage_start = time.time()
            labeled_transcript = self.apply_labels(
                transcript, speaker_segments, speaker_labels
            )
            stage3_time = time.time() - stage_start
            stage_timings["stage3"] = stage3_time

            stage3_details = [f"Labeled {len(labeled_transcript)} transcript segments"]
            stage_details["stage3"] = {"status": "success", "details": stage3_details}

            self._display_stage_completion(
                3, 5, "Label application", stage3_time, stage3_details
            )

            # Step 4: QA assessment
            stage_start = time.time()
            qa_result = self.qa_llm.assess_qa(labeled_transcript)
            stage4_time = time.time() - stage_start
            stage_timings["stage4"] = stage4_time

            stage4_details = [
                f"Overall Rating: {qa_result['overall_rating']['score']}/10"
            ]
            stage_details["stage4"] = {"status": "success", "details": stage4_details}

            self._display_stage_completion(
                4, 5, "QA assessment", stage4_time, stage4_details
            )

            # Step 5: Build output
            stage_start = time.time()
            output = self.build_output(
                audio_path,
                start_time,
                transcript,
                speaker_segments,
                speaker_labels,
                labeled_transcript,
                qa_result,
            )

            # Save output
            output_path = self._save_output(audio_path, output)
            stage5_time = time.time() - stage_start
            stage_timings["stage5"] = stage5_time

            stage5_details = [f"Output saved: {output_path}"]
            stage_details["stage5"] = {"status": "success", "details": stage5_details}

            self._display_stage_completion(
                5, 5, "Building output", stage5_time, stage5_details
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Complete ({processing_time:.1f}s)")

            return output

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()

            # Determine which stage failed based on what's in stage_timings
            failed_stage_num = len(stage_timings) + 1
            failed_stage_time = time.time() - stage_start

            stage_names = {
                1: "Audio processing",
                2: "Participant labeling",
                3: "Label application",
                4: "QA assessment",
                5: "Building output",
            }

            failed_stage_name = stage_names.get(failed_stage_num, "Unknown stage")

            # Build context from completed stages
            context_details = []
            for stage_num in range(1, failed_stage_num):
                stage_info = stage_details.get(f"stage{stage_num}", {})
                if stage_info.get("status") == "success":
                    timing = stage_timings.get(f"stage{stage_num}", 0)
                    stage_name = stage_names.get(
                        stage_num,
                    )
                    context_details.append(f"{stage_name}: ✅ ({timing:.1f}s)")

            context_details.append(
                f"{failed_stage_name}: ⏳ (failed after {failed_stage_time:.1f}s)"
            )

            self._display_stage_failure(
                failed_stage_num,
                5,
                failed_stage_name,
                failed_stage_time,
                str(e),
                [f"File: {filename}"] + context_details,
            )

            raise RuntimeError(f"Pipeline processing failed for {filename}: {e}")

    def apply_labels(self, transcript, speaker_segments, speaker_labels):
        """
        Apply participant labels to transcript segments.

        Args:
            transcript (list): List of transcript segments
            speaker_segments (list): List of speaker segments with timestamps
            speaker_labels (dict): Mapping of speaker IDs to roles

        Returns:
            list: Labeled transcript segments with roles
        """
        labeled_transcript = []

        for transcript_segment in transcript:
            segment_start = transcript_segment.get("start", 0)
            segment_end = transcript_segment.get("end", 0)

            # Find which speaker segment this transcript segment belongs to
            speaker_id = self._find_speaker_for_segment(
                segment_start, segment_end, speaker_segments
            )

            # Apply label
            role = speaker_labels.get(speaker_id, {}).get("role", "unknown")

            labeled_segment = {
                "speaker_id": speaker_id,
                "role": role,
                "start": segment_start,
                "end": segment_end,
                "text": transcript_segment["text"],
            }

            labeled_transcript.append(labeled_segment)

        return labeled_transcript

    def build_output(
        self,
        audio_path,
        start_time,
        transcript,
        speaker_segments,
        speaker_labels,
        labeled_transcript,
        qa_result,
    ):
        """
        Build final JSON output with all required fields.

        Args:
            audio_path (str): Path to audio file
            start_time (datetime): Processing start time
            transcript (list): Transcript segments
            speaker_segments (list): Speaker segments
            speaker_labels (dict): Speaker labels
            labeled_transcript (list): Labeled transcript
            qa_result (dict): QA assessment result

        Returns:
            dict: Complete JSON output
        """
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Build participants list
        participants = self._build_participants_list(
            transcript, speaker_segments, speaker_labels
        )

        # Build output structure
        output = {
            "metadata": {
                "filename": Path(audio_path).name,
                "processed_at": end_time.isoformat(),
                "processing_time_seconds": round(processing_time, 2),
            },
            "participants": participants,
            "qa_assessment": {
                "resolution_quality": {
                    "score": qa_result["resolution_quality"]["score"],
                    "reasoning": qa_result["resolution_quality"]["reasoning"],
                },
                "tone_phenomena": {
                    "score": qa_result["tone_phenomena"]["score"],
                    "reasoning": qa_result["tone_phenomena"]["reasoning"],
                },
                "compliance": {
                    "score": qa_result["compliance"]["score"],
                    "reasoning": qa_result["compliance"]["reasoning"],
                },
                "overall_rating": {
                    "score": qa_result["overall_rating"]["score"],
                    "reasoning": qa_result["overall_rating"]["reasoning"],
                },
            },
            "call_analysis": {
                "reason": qa_result["call_reason"],
                "category": qa_result["category"],
                "summary": qa_result["summary"],
                "strengths": qa_result["strengths"],
                "improvements": qa_result["improvements"],
            },
            "full_transcript": [
                {
                    "speaker_id": seg["speaker_id"],
                    "role": seg["role"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in labeled_transcript
            ],
        }

        return output

    def _save_output(self, audio_path, output):
        """
        Save output JSON to file.

        Args:
            audio_path (str): Original audio file path
            output (dict): Output data to save

        Returns:
            str: Path to saved output file
        """
        # Generate filename with timestamp
        audio_name = Path(audio_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{audio_name}_{timestamp}.json"
        output_path = os.path.join(self.output_directory, output_filename)

        # Save with formatting
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return output_path

    def _find_speaker_for_segment(self, segment_start, segment_end, speaker_segments):
        """
        Find which speaker segment a transcript segment belongs to.

        Args:
            segment_start (float): Transcript segment start time
            segment_end (float): Transcript segment end time
            speaker_segments (list): List of speaker segments

        Returns:
            str: Speaker ID
        """
        # Find the speaker segment that has the most overlap with this transcript segment
        best_speaker_id = "UNKNOWN"
        max_overlap = 0

        for speaker_segment in speaker_segments:
            speaker_start = speaker_segment["start"]
            speaker_end = speaker_segment["end"]
            speaker_id = speaker_segment["speaker_id"]

            # Calculate overlap
            overlap_start = max(segment_start, speaker_start)
            overlap_end = min(segment_end, speaker_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker_id = speaker_id

        return best_speaker_id

    def _build_participants_list(self, transcript, speaker_segments, speaker_labels):
        """
        Build participants list with segments grouped by speaker.

        Args:
            transcript (list): Transcript segments
            speaker_segments (list): Speaker segments
            speaker_labels (dict): Speaker labels

        Returns:
            list: Participants with their segments
        """
        participants = {}

        for segment in speaker_segments:
            speaker_id = segment["speaker_id"]
            role = speaker_labels.get(speaker_id, {}).get("role", "unknown")
            name = speaker_labels.get(speaker_id, {}).get("name")

            if speaker_id not in participants:
                participants[speaker_id] = {
                    "speaker_id": speaker_id,
                    "role": role,
                    "name": name,
                    "segments": [],
                }

        for segment in transcript:
            speaker_id = self._find_speaker_for_segment(
                segment.get("start", 0), segment.get("end", 0), speaker_segments
            )

            role = speaker_labels.get(speaker_id, {}).get("role", "unknown")
            name = speaker_labels.get(speaker_id, {}).get("name")

            if speaker_id not in participants:
                participants[speaker_id] = {
                    "speaker_id": speaker_id,
                    "role": role,
                    "name": name,
                    "segments": [],
                }

            participants[speaker_id]["segments"].append(
                {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", ""),
                }
            )

        return list(participants.values())

    def _create_speaker_segments_from_transcript(self, transcript):
        """
        Create speaker segments from transcript when diarization is unavailable.

        Args:
            transcript (list): Transcript segments

        Returns:
            list: Speaker segments
        """
        # Simple approach: assume alternating speakers
        speaker_segments = []
        for i, segment in enumerate(transcript):
            speaker_segments.append(
                {
                    "speaker_id": f"SPEAKER_{i % 2}",
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                }
            )
        return speaker_segments
