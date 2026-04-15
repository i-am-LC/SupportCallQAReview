# Support Call QA System

A fully private support call quality assurance system that processes audio recordings locally, transcribes them, identifies participants, and provides structured QA assessments - all with low hardware requirements (8GB VRAM).

## Purpose

- **Fully Private**: All processing happens locally - no external API calls beyond model downloads
- **Low Hardware Requirements**: Optimized to run on 8GB VRAM GPU
- **OpenAI-Compatible LLM**: Uses any HTTP endpoint supporting OpenAI API format
- **Structured Output**: Generates detailed JSON with QA scores, participant roles, and commentary

## Prerequisites

### Required

1. **Python 3.9+ with pip**
   - Required for virtual environment and package management
   - Install Python: https://www.python.org/downloads/

2. **CUDA-compatible GPU with 8GB VRAM**
   - Required for fast-whisper and pyannote.audio processing
   - Tested on NVIDIA GPUs with CUDA support

3. **HuggingFace Account**
   - Required for pyannote.audio model access
   - Get token from https://huggingface.co/settings/tokens

4. **OpenAI-Compatible HTTP Endpoint**
   - Must be already running and accessible
   - Examples of compatible providers:
     - [llama.cpp](https://github.com/ggerganov/llama.cpp) HTTP server
     - [vLLM](https://github.com/vllm-project/vllm) OpenAI API
     - [Ollama](https://ollama.com/) API
   - Must support `/v1/completions` and `/v1/chat/completions` endpoints
   - Must include models specified in your config.json

### Typical Endpoint Configuration

```json
{
  "llm_provider": {
    "base_url": "http://localhost:8080/v1",    // llama.cpp default
    // or
    "base_url": "http://localhost:11434/v1",   // Ollama default
    // or
    "base_url": "http://localhost:8000/v1"     // vLLM default
  }
}
```

## Installation

1. Clone the repository or navigate to your project directory

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Get HuggingFace token:
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the model terms
   - Get token from https://huggingface.co/settings/tokens

5. Create configuration files:
```bash
cp .env.example .env
cp config.json config.json
```

6. Configure credentials in `.env`:
```bash
# HuggingFace token
HF_TOKEN=your_huggingface_token_here

# FTP credentials (if using FTP)
FTP_IP_ADDRESS=your.ftp.server
FTP_USERNAME=your_username
FTP_PASSWORD=your_password
```

7. Setup HTTP LLM endpoint:
   Start your OpenAI-compatible HTTP endpoint (llama.cpp, vLLM, or Ollama) and verify it matches the URL in `config.json`.

8. Run your first test:
```bash
# Place a .wav file in input/ directory, then run:
python main.py --source local --config config.json --verbose

# Results appear in output/ directory as {recording_name}_{timestamp}.json
```

## Configuration

### 1. Environment Variables (.env)

Configure your environment variables in `.env`:

```bash
# HuggingFace token for pyannote.audio model download
HF_TOKEN=your_huggingface_token_here

# FTP credentials (if FTP enabled)
FTP_IP_ADDRESS=your.ftp.server
FTP_USERNAME=your_username
FTP_PASSWORD=your_password

# Optional: Override LLM endpoint
# LLM_BASE_URL=http://localhost:8080/v1
```

### 2. Main Configuration (config.json)

Update `config.json` with your settings:

```json
{
  "directories": {
    "input": "input/",
    "output": "output/"
  },
  
  "ftp": {
    "enabled": true,  // Set to false to skip FTP pulls
    "download_directory": "input/"
    // Note: FTP credentials come from .env file
  },
  
  "llm_provider": {
    "base_url": "http://localhost:8080/v1",
    "api_key": "dummy",
    "models": {
      "participant_labeler": "gemma-2-4b-q4",    // Model for role labeling and name identification
      "qa_assessor": "qwen2.5-4b-q4"             // Model for QA assessment
    },
    "parameters": {
      "temperature": 0.0,
      "max_tokens": 2048
    }
  },
  
"audio_processing": {
  "whisper": {
    "model": "small",
    "language": "en",
    "device": "cuda",
    "compute_type": "int8",
    "initial_prompt": "Support agent names: Sarah, John, Mike, Lisa, David..."
  }
}
```

#### Language Setting

The `language` option in whisper configuration:

```json
"language": "en"  // Force English transcription
```

Valid values:
- `"en"` - Force English (faster, more accurate for English audio)
- `null` or omit - Auto-detect language

For auto-detection, simply omit the language field or set to `null`.

#### Initial Prompt for Transcription

The `initial_prompt` option provides context to faster-whisper to improve transcription accuracy for domain-specific terms and names:

```json
"initial_prompt": "Support agent names: Sarah, John, Mike, Lisa, David..."
```

This helps the model:
- Recognise common support agent names
- Understand domain-specific terminology (account, ticket, billing, etc.)
- Improve accuracy foracronyms and abbreviations

You can customize this prompt in `config.json` to match your organization's specific needs.

## Usage

**Important:** Ensure your virtual environment is activated before running any commands:

```bash
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Basic Usage

Process local audio files:

```bash
python main.py --source local --config config.json
```

### Unified Processing

Download from FTP and process in one command:

```bash
python main.py --source ftp --ftp-date 20260413 --config config.json
```

This will:
1. Clear input/ and output/ directories (unless `--no-clean` is used)
2. Download audio files from the specified FTP date
3. Process each file through the complete pipeline
4. Generate JSON output files

### Options

```bash
# List available FTP date directories
python main.py --list-ftp-dates

# Download from single date directory
python main.py --source ftp --ftp-date 20260413 --config config.json

# Download from date range
python main.py --source ftp --ftp-date-start 20260410 --ftp-date-end 20260413 --config config.json

# Process only local files
python main.py --source local --config config.json

# Override input directory
python main.py --source local --input-dir /path/to/other/input

# Use different config file
python main.py --config path/to/other_config.json

# Enable verbose logging
python main.py --source local --config config.json --verbose

# Skip GPU check (not recommended)
python main.py --source local --config config.json --no-gpu-check

# Skip directory clearing (keep existing files)
python main.py --source local --config config.json --no-clean
```

### CLI Arguments

- `--config`: Path to configuration file (default: `config.json`)
- `--source`: Source of audio files - `ftp`, `local`, or `both` (default: `both`)
- `--ftp-date`: Single date directory (YYYYMMDD format). Not compatible with date range
- `--ftp-date-start`: Start date for range (YYYYMMDD format). Requires --ftp-date-end
- `--ftp-date-end`: End date for range (YYYYMMDD format). Requires --ftp-date-start
- `--list-ftp-dates`: List available date directories on FTP server
- `--input-dir`: Override input directory from config (optional)
- `--no-gpu-check`: Skip GPU availability check (not recommended)
- `--no-clean`: Skip clearing input/output directories
- `--verbose`: Enable verbose logging for debugging

### Report Generation

After batch processing completes, the system prompts:

```bash
Generate support rep report? (y/n): y
```

Select `y` to generate a markdown report containing:
- Up to 4 random calls per support representative
- Full transcripts with speaker names
- QA assessment scores and reasoning
- Call analysis (summary, strengths, improvements)

Reports are saved to `output/support_rep_report_{timestamp}.md`.

### File Organization

- ✅ `main.py` - Entry point with CLI arguments
- ✅ `src/pipeline.py` - Complete workflow orchestration
- ✅ `src/ftp_fetcher.py` - Automated FTP download
- ✅ `src/audio_processor.py` - Transcription and diarization
- ✅ `src/participant_llm.py` - Role labeling via HTTP LLM
- ✅ `src/qa_llm.py` - QA assessment via HTTP LLM
- ✅ `src/models.py` - Pydantic data models
- ✅ `src/report_generator.py` - Markdown report generation


### FTP Configuration

To enable automatic audio file download from FTP:

1. Set FTP credentials in `.env`:
```bash
FTP_IP_ADDRESS=your.ftp.server
FTP_USERNAME=your_username
FTP_PASSWORD=your_password
```

2. Enable FTP in `config.json`:
```json
{
  "ftp": {
    "enabled": true,
    "download_directory": "input/"
  }
}
```

3. Use date filtering with CLI:
```bash
# List available date directories
python main.py --list-ftp-dates

# Download from single date directory
python main.py --source ftp --ftp-date 20260413

# Download from date range
python main.py --source ftp --ftp-date-start 20260410 --ftp-date-end 20260413

# Download and process
python main.py --source ftp --ftp-date 20260413 --config config.json
```

**Note:** Date specification is REQUIRED when `--source ftp` is specified. Choose either:
- **Single date mode**: `--ftp-date YYYYMMDD`
- **Date range mode**: `--ftp-date-start YYYYMMDD --ftp-date-end YYYYMMDD`

**FTP Features:**
- ✅ Single date directory download OR date range download
- ✅ Date format validation (YYYYMMDD only)
- ✅ Calendar date validation (real dates only)
- ✅ Date range validation (start <= end, inclusive)
- ✅ Error if date directory/directories don't exist
- ✅ List available directories with `--list-ftp-dates`
- ✅ Skips files that already exist locally
- ✅ Downloads to configurable directory (`input/` by default)
- ✅ No interactive prompts - fully automated
- ✅ Support for both single and date range modes

## Output Format

Processed calls are saved as JSON files in the `output/` directory with format: `{recording_name}_{timestamp}.json`

### Example Output Structure

```json
{
  "metadata": {
    "filename": "recording_001.wav",
    "processed_at": "2026-04-13T14:30:00Z",
    "processing_time_seconds": 45.2
  },
  "participants": [
    {
      "speaker_id": "SPEAKER_00",
      "role": "agent",
      "name": "Sarah",
      "segments": [
        {
          "start": 0.0,
          "end": 5.2,
          "text": "Hello, thank you for calling support, this is Sarah."
        }
      ]
    },
    {
      "speaker_id": "SPEAKER_01",
      "role": "customer",
      "name": null,
      "segments": [
        {
          "start": 5.5,
          "end": 8.1,
          "text": "Hi, I'm having an issue with my account."
        }
      ]
    }
  ],
  "qa_assessment": {
    "resolution_quality": {
      "score": 4,
      "reasoning": "Agent successfully resolved the issue without escalation."
    },
    "tone_phenomena": {
      "score": 5,
      "reasoning": "Professional and empathetic throughout. No negative sentiment detected."
    },
    "compliance": {
      "score": 5,
      "reasoning": "No data breaches or policy violations. Followed GDPR protocols."
    },
    "overall_rating": {
      "score": 9,
      "reasoning": "Excellent call with clear resolution and professional demeanor."
    }
  },
  "call_analysis": {
    "reason": "Customer unable to access account after password reset",
    "category": "Technical Support",
    "summary": "Agent walked customer through account recovery process in clear, step-by-step manner.",
    "strengths": [
      "Clear communication",
      "Efficient resolution"
    ],
    "improvements": [
      "Could offer proactive tips for future issues",
      "Follow-up call recommended"
    ]
  },
  "full_transcript": [
    {
      "speaker_id": "SPEAKER_00",
      "role": "agent",
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, thank you for calling support."
    },
    {
      "speaker_id": "SPEAKER_01",
      "role": "customer",
      "start": 5.5,
      "end": 8.1,
      "text": "Hi, I'm having an issue with my account."
    }
  ]
}
```

## Processing Pipeline

### Complete Workflow

1. **Audio Download** (if FTP enabled)
    - Automatic FTP connection using credentials from `.env`
    - Validates specified date range OR single directory exists on server
    - Downloads audio files from specified date range or single directory
    - Saves to configured `download_directory` (default: `input/`)
    - Skips files that already exist locally
    - Date format must be exactly YYYYMMDD
    - Date range includes both start and end dates (inclusive)

2. **Audio Processing** (`AudioProcessor`)
   - fast-whisper (Turbo model) generates transcript with timestamps
   - pyannote.audio 3.1 identifies speaker segments
   - Returns structured data: transcript segments and speaker identification

3. **Participant Labeling** (`ParticipantLabeler`)
   - HTTP call to configured LLM endpoint
   - Analyzes full transcript and speaker segments
   - Identifies which speakers are agents and which are customers
   - Extracts participant names
   - Uses LangChain structured output with `ParticipantLabels` model
   - Normalizes speaker IDs to match pyannote format (SPEAKER_0 -> SPEAKER_00)
   - Returns: `{"labels": {"SPEAKER_00": {"role": "agent", "name": "Sarah"}, ...}}`

4. **Label Application** (`Pipeline.apply_labels()`)
   - Maps speaker IDs to agent/customer roles and names
   - Applies roles and names to all transcript segments
   - Aligns transcript segments with speaker segments based on timestamps
   - Returns labeled transcript with roles and names

5. **QA Assessment** (`QAAssessor`)
   - HTTP call to configured LLM endpoint
   - Analyzes labeled transcript with agent/customer roles
   - Uses LangChain structured output with `QAAssessment` model
   - Rates 4 criteria:
     - Resolution Quality (0-5)
     - Tone (0-5)
     - Compliance (0-5)
     - Overall Rating (0-10)
   - Provides summary, strengths, improvements
   - Identifies call category and reason
   - Returns structured QA data with all fields

6. **JSON Output** (`Pipeline.build_output()`)
   - Combines all data into structured JSON
   - Includes: metadata, participants, QA assessment, call analysis, full transcript
   - Validates against schema defined in config.json

7. **Report Generation** (interactive)
   - Prompts user after processing completes
   - Generates markdown report with up to 4 random calls per agent
   - Report includes: transcripts, QA scores, call analysis
   - Saved to `output/support_rep_report_{timestamp}.md`
   - Saves to `output/` directory with timestamped filename

### Pipeline Components

The pipeline orchestrates four main components:

1. **AudioProcessor**: Handles transcription and speaker diarization
2. **ParticipantLabeler**: Identifies agent/customer roles via HTTP LLM
3. **QAAssessor**: Provides QA assessment via HTTP LLM
4. **Pipeline**: Orchestrates complete workflow and builds output

All components are initialized with configuration from `config.json` and process data sequentially to optimize memory usage.

## Troubleshooting

### Virtual Environment Issues

**Issue:** Command not found or module import errors

**Solution:**
- Ensure virtual environment is activated:
  ```bash
  # On Linux/Mac:
  source venv/bin/activate
  # On Windows:
  venv\Scripts\activate
  ```
- Verify pip is installed in venv: `which pip` or `where pip`
- Reinstall dependencies if needed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should show Python from venv)
- For PyTorch compatibility issues, see PyTorch version note below

### PyTorch Version Compatibility

**Issue:** Model loading errors with pyannote.audio

**Solution:**
- This project requires PyTorch 2.5.x (not 2.6+)
- Install compatible versions:
  ```bash
  pip install torch==2.5.1 torchaudio==2.5.1
  pip install huggingface_hub==0.20.3
  ```

### GPU Not Available

**Error:** `ERROR: GPU required but not available`

**Solution:**
- Ensure NVIDIA drivers are installed
- Verify CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Check that CUDA version is compatible with PyTorch version

### HuggingFace Token Issues

**Error:** `Cannot access gated model "pyannote/speaker-diarization-3.1"`

**Solution:**
- Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the model terms
- Get token from https://huggingface.co/settings/tokens
- Verify your token is correctly set in `.env`

### HTTP Endpoint Connection Errors

**Error:** `Connection refused` or `Failed to connect to LLM endpoint`

**Solution:**
- Verify your LLM HTTP server is running
- Check that `base_url` in config.json matches your endpoint
- Test endpoint manually with curl:
  ```bash
  curl http://localhost:8080/v1/models
  ```
- Ensure the models specified in config.json are available at your endpoint

### Memory Issues

**Error:** `CUDA out of memory`

**Solution:**
- Ensure you have at least 8GB VRAM
- Try reducing batch size in config.json
- Ensure no other GPU processes are running
- Consider using Whisper `small` model instead of `turbo`

### Fast-Whisper vs Pyannote Speaker ID Mismatch

**Issue**: Different speaker IDs between transcription and diarization

**Solution:**
- The pipeline automatically aligns segments based on timestamps
- Check that audio sample rate is 16kHz
- Verify audio file integrity

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (CUDA-compatible)
- **RAM**: 16GB recommended
- **Storage**: Enough space for audio files and output JSON

## Testing

To verify your setup is working correctly, you can run the HTTP LLM integration test:

```bash
python test_llm_integration.py
```

This test will:
1. Verify the ParticipantLabeler module can connect to your LLM endpoint
2. Verify the QAAssessor module can connect to your LLM endpoint
3. Validate JSON response parsing
4. Ensure structured outputs are correctly formatted

**Note:** The test requires your LLM HTTP endpoint to be running and accessible. Check your endpoint configuration in `config.json` before running tests.

For end-to-end testing with real audio files:
1. Place a `.wav` file in the `input/` directory
2. Ensure all prerequisites are met (GPU, HF token, HTTP endpoint)
3. Run: `python main.py --source local --config config.json`
4. Check the `output/` directory for the JSON result

## Privacy & Security

- **All processing is local** - no audio or transcript data leaves your machine
- **HuggingFace token** is only used for model download, not inference
- **No external API calls** beyond initial model downloads
- **Output files** remain on your local system

## License

See LICENSE file for details.

## Support

For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review config.json settings
- Verify all prerequisites are met