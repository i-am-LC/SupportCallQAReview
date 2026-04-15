import os
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

load_dotenv()


class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.device = config["audio_processing"]["whisper"]["device"]
        self.language = config["audio_processing"]["whisper"].get("language", None)

        self.whisper_model = self._load_whisper_model()
        self.diarizer = self._load_pyannote_pipeline()

    def _load_whisper_model(self):
        whisper_config = self.config["audio_processing"]["whisper"]

        device = whisper_config["device"]
        compute_type = whisper_config.get("compute_type", "int8")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required but not available. "
                "This system requires a CUDA-compatible GPU with 8GB VRAM."
            )

        print(f"Loading faster-whisper model: {whisper_config['model']} on {device}")
        model = WhisperModel(
            whisper_config["model"], device=device, compute_type=compute_type
        )

        return model

    def _load_pyannote_pipeline(self):
        diarization_config = self.config["audio_processing"]["diarization"]

        if not diarization_config.get("use_hf_token", False):
            raise RuntimeError(
                "pyannote.audio requires HF token, but use_hf_token is False. "
                "Set use_hf_token to true or disable diarization."
            )

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not found in environment variables. "
                "Set HF_TOKEN in .env file to enable pyannote.audio."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA required but not available. "
                "This system requires a CUDA-compatible GPU with 8GB VRAM."
            )

        print(f"Loading pyannote.audio pipeline: {diarization_config['model']}")
        try:
            pipeline = Pipeline.from_pretrained(
                diarization_config["model"], use_auth_token=hf_token
            )

            pipeline = pipeline.to(torch.device("cuda"))
            print("Diarization pipeline moved to CUDA")

            return pipeline
        except Exception as e:
            raise RuntimeError(
                f"ERROR loading pyannote.audio: {e}. "
                "Ensure HF_TOKEN is valid and you have accepted the model terms."
            )

    def process(self, audio_path):
        result = {"transcript": [], "speaker_segments": []}

        # Step 1: Transcribe audio with faster-whisper
        print(f"Transcribing: {audio_path}")
        segments, info = self.whisper_model.transcribe(
            audio_path,
            language=self.language,
            task="transcribe",
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=self.config["audio_processing"]["whisper"].get(
                "initial_prompt"
            ),
        )

        # Convert segments to list and store
        transcript_segments = []
        for segment in segments:
            transcript_segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )

        result["transcript"] = transcript_segments
        print(f"Transcription complete: {len(transcript_segments)} segments")

        # Step 2: Diarize speakers with pyannote.audio
        if self.diarizer is not None:
            print(f"Diarizing: {audio_path}")
            diarization = self.diarizer(audio_path)

            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append(
                    {"speaker_id": speaker, "start": turn.start, "end": turn.end}
                )

            result["speaker_segments"] = speaker_segments
            print(f"Diarization complete: {len(speaker_segments)} speaker segments")
        else:
            print("Skipping diarization (pipeline not available)")
            result["speaker_segments"] = []

        return result
