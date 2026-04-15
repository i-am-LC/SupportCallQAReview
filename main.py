import sys
import os
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ftplib

# Import pipeline components
from src.pipeline import Pipeline
from src.ftp_fetcher import FTPFetcher

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("support_call_qa.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Suppress verbose library warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*TF32.*")
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*weights_only.*")


class ProgressTracker:
    """Track overall processing progress and provide time estimates."""

    def __init__(self, total_files):
        self.total_files = total_files
        self.completed_files = 0
        self.start_time = time.time()
        self.total_processing_time = 0.0

    def update(self, processing_time):
        """Update progress tracker with completed file processing time."""
        self.completed_files += 1
        self.total_processing_time += processing_time

    def get_progress(self):
        """Calculate current progress percentage and estimated remaining time."""
        if self.total_files == 0:
            return 0.0, 0.0

        percentage = (self.completed_files / self.total_files) * 100

        if self.completed_files > 0:
            avg_time = self.total_processing_time / self.completed_files
            remaining_files = self.total_files - self.completed_files
            est_remaining = remaining_files * avg_time
        else:
            est_remaining = 0.0

        return percentage, est_remaining

    def get_display_info(self):
        """Get formatted progress information for display."""
        percentage, est_remaining = self.get_progress()
        progress_bar = _generate_progress_bar(self.completed_files, self.total_files)
        est_time_str = _format_time(est_remaining)
        return progress_bar, percentage, est_time_str


def _generate_progress_bar(completed, total):
    """Generate hash-based progress bar."""
    if total == 0:
        return "[#-------------------] 0/0 (0%)"

    filled = int((completed / total) * 20)  # 20 hash marks for 100%
    empty = 20 - filled
    bar = f"[{'#' * filled}{'-' * empty}]"
    percentage = (completed / total) * 100
    return f"{bar} {completed}/{total} ({percentage:.0f}%)"


def _format_time(seconds):
    """Format seconds to readable time format."""
    if seconds < 60:
        return f"~{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"~{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        return f"~{hours}h {minutes}m"


def substitute_env_vars(config):
    """
    Substitute environment variables in config values.

    Supports ${VAR_NAME} syntax in string values.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Configuration with substituted values
    """
    if isinstance(config, str):
        # Check for environment variable pattern
        if config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            value = os.getenv(var_name)
            if value is None:
                logger.warning(
                    f"Environment variable {var_name} not found, keeping original value"
                )
                return config
            return value
        return config
    elif isinstance(config, dict):
        return {key: substitute_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    else:
        return config


def load_config(config_path):
    """
    Load and validate configuration file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Substitute environment variables
        config = substitute_env_vars(config)

        # Validate required sections
        required_sections = ["directories", "llm_provider", "audio_processing"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Configuration missing required section: {section}")

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def check_gpu_availability():
    """
    Check if CUDA GPU is available.

    Returns:
        bool: True if GPU available, False otherwise
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("CUDA required but not available")
            logger.error("This system requires a CUDA-compatible GPU with 8GB VRAM")
            return False

        # Log GPU info
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✓ GPU detected: {gpu_name}")
        logger.info(f"✓ GPU count: {gpu_count}")

        return True

    except ImportError:
        logger.error("PyTorch not installed. Cannot check GPU availability")
        return False
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return False


def validate_ftp_date_range(ftp_date, ftp_date_start, ftp_date_end):
    """
    Validate FTP date inputs (single date OR date range - Option B).

    Args:
        ftp_date (str): Single date from CLI
        ftp_date_start (str): Start date from CLI
        ftp_date_end (str): End date from CLI

    Returns:
        tuple: (mode, date_or_range) where mode is 'single' or 'range'

    Raises:
        RuntimeError: If invalid combination or invalid dates
    """
    import re

    # Case 1: Single date mode
    if ftp_date and not (ftp_date_start or ftp_date_end):
        # Only --ftp-date provided
        if not re.match(r"^\d{8}$", ftp_date):
            raise RuntimeError(
                f"Invalid FTP date format: {ftp_date}\n"
                f"Expected format: YYYYMMDD (exactly 8 digits)\n"
                f"Example: 20260413"
            )

        # Validate it's a real calendar date
        try:
            datetime.strptime(ftp_date, "%Y%m%d")
        except ValueError:
            raise RuntimeError(
                f"Invalid FTP date: {ftp_date}\n"
                f"Must be a valid calendar date in YYYYMMDD format"
            )

        return ("single", ftp_date)

    # Case 2: Date range mode
    elif ftp_date_start and ftp_date_end:
        # Both start and end provided
        if not re.match(r"^\d{8}$", ftp_date_start):
            raise RuntimeError(
                f"Invalid FTP date-start format: {ftp_date_start}\n"
                f"Expected format: YYYYMMDD (exactly 8 digits)\n"
                f"Example: 20260410"
            )

        if not re.match(r"^\d{8}$", ftp_date_end):
            raise RuntimeError(
                f"Invalid FTP date-end format: {ftp_date_end}\n"
                f"Expected format: YYYYMMDD (exactly 8 digits)\n"
                f"Example: 20260415"
            )

        # Validate both are real calendar dates
        try:
            start_dt = datetime.strptime(ftp_date_start, "%Y%m%d")
        except ValueError:
            raise RuntimeError(
                f"Invalid FTP date-start: {ftp_date_start}\n"
                f"Must be a valid calendar date in YYYYMMDD format"
            )

        try:
            end_dt = datetime.strptime(ftp_date_end, "%Y%m%d")
        except ValueError:
            raise RuntimeError(
                f"Invalid FTP date-end: {ftp_date_end}\n"
                f"Must be a valid calendar date in YYYYMMDD format"
            )

        # Validate start <= end
        if start_dt > end_dt:
            raise RuntimeError(
                f"Start date ({ftp_date_start}) must be before or equal to end date ({ftp_date_end})"
            )

        return ("range", (ftp_date_start, ftp_date_end))

    # Case 3: No date provided
    elif not ftp_date and not (ftp_date_start or ftp_date_end):
        raise RuntimeError(
            "FTP date required for --source ftp\n"
            "Choose one of:\n"
            "  - Single date: --ftp-date YYYYMMDD\n"
            "  - Date range: --ftp-date-start YYYYMMDD --ftp-date-end YYYYMMDD"
        )

    # Case 4: Invalid combination
    else:
        if ftp_date and (ftp_date_start or ftp_date_end):
            raise RuntimeError(
                "Cannot specify both --ftp-date and date range\n"
                "Use either:\n"
                "  - Single date: --ftp-date YYYYMMDD\n"
                "  - Date range: --ftp-date-start YYYYMMDD --ftp-date-end YYYYMMDD"
            )
        else:
            raise RuntimeError(
                "Incomplete date range specification\n"
                "Requires both --ftp-date-start and --ftp-date-end\n"
                "Example: --ftp-date-start 20260410 --ftp-date-end 20260413"
            )


def clear_directories(config):
    """
    Clear input and output directories for a clean slate.

    Args:
        config (dict): Configuration dictionary
    """
    input_dir = config["directories"]["input"]
    output_dir = config["directories"]["output"]

    print("=" * 70)
    print("Clearing directories for clean slate...")
    print("=" * 70)

    # Clear input directory
    input_path = Path(input_dir)
    if input_path.exists():
        input_files = list(input_path.glob("*"))
        for file in input_files:
            if file.is_file():
                file.unlink()
        print(f"✓ Cleared {len(input_files)} files from {input_dir}")
    else:
        os.makedirs(input_dir, exist_ok=True)
        print(f"✓ Created input directory: {input_dir}")

    # Clear output directory
    output_path = Path(output_dir)
    if output_path.exists():
        output_files = list(output_path.glob("*"))
        for file in output_files:
            if file.is_file():
                file.unlink()
        print(f"✓ Cleared {len(output_files)} files from {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")

    print("=" * 70)


def confirm_directory_clearing(config, clear=False):
    """
    Clear directories based on --clear flag.

    Args:
        config (dict): Configuration dictionary
        clear (bool): If True, clear directories; if False, skip

    Returns:
        bool: True if directories were cleared, False if skipped
    """
    if not clear:
        logger.info("Skipping directory clearing (use --clear to enable)")
        return False

    input_dir = config["directories"]["input"]
    output_dir = config["directories"]["output"]

    input_path = Path(input_dir)
    if input_path.exists():
        input_count = len([f for f in input_path.glob("*") if f.is_file()])

    output_path = Path(output_dir)
    if output_path.exists():
        output_count = len([f for f in output_path.glob("*") if f.is_file()])

    print(f"\ninput/: {input_count} files")
    print(f"output/: {output_count} files")

    clear_directories(config)
    return True


def list_ftp_directories(config):
    """
    List available date directories on FTP server.

    Args:
        config (dict): Configuration dictionary

    Raises:
        RuntimeError: If FTP connection fails
    """
    import re

    ftp_config = config.get("ftp", {})

    if not ftp_config.get("enabled", False):
        logger.info("FTP fetch disabled in configuration")
        return

    logger.info("Connecting to FTP server...")

    ip_address = os.getenv("FTP_IP_ADDRESS")
    username = os.getenv("FTP_USERNAME")
    password = os.getenv("FTP_PASSWORD")

    if not all([ip_address, username, password]):
        raise RuntimeError(
            "FTP credentials required\n"
            "Set FTP_IP_ADDRESS, FTP_USERNAME, FTP_PASSWORD in .env"
        )

    try:
        ftp = ftplib.FTP(ip_address)
        ftp.login(username=username, passwd=password)
        logger.info(f"Connected to FTP server: {ip_address}")

        # Get all directories
        directories = []
        for item in ftp.nlst():
            try:
                current = ftp.pwd()
                if item not in [".", ".."]:
                    ftp.cwd(item)
                    directories.append(item)
                    ftp.cwd("..")
            except ftplib.error_perm:
                pass

        # Filter to YYYYMMDD directories only
        date_dirs = [d for d in directories if re.match(r"^\d{8}$", d)]

        date_dirs.sort()

        logger.info("=" * 70)
        logger.info("Available Date Directories (YYYYMMDD format):")
        logger.info("=" * 70)

        if date_dirs:
            for date_dir in date_dirs:
                logger.info(f"  {date_dir}")
        else:
            logger.warning("No date directories found")

        logger.info("=" * 70)
        logger.info(f"Total: {len(date_dirs)} directories")

        ftp.quit()

    except Exception as e:
        logger.error(f"Failed to list FTP directories: {e}")
        raise


def fetch_from_ftp(config, ftp_date=None, ftp_date_start=None, ftp_date_end=None):
    """
    Fetch audio files from FTP server (single date or date range).

    Args:
        config (dict): Configuration dictionary
        ftp_date (str): Single date (YYYYMMDD)
        ftp_date_start (str): Start date for range (YYYYMMDD)
        ftp_date_end (str): End date for range (YYYYMMDD)

    Returns:
        list: List of downloaded file paths

    Raises:
        RuntimeError: If dates are invalid or directory doesn't exist
    """
    ftp_config = config.get("ftp", {})

    if not ftp_config.get("enabled", False):
        logger.info("FTP fetch disabled in configuration")
        return []

    # Validate date inputs (single OR range)
    mode, date_value = validate_ftp_date_range(ftp_date, ftp_date_start, ftp_date_end)

    # Add to config based on mode
    if mode == "single":
        ftp_config["date"] = date_value
        logger.info(f"Fetching files from FTP date directory: {date_value}")
    else:  # mode == 'range'
        ftp_config["date_start"] = date_value[0]
        ftp_config["date_end"] = date_value[1]
        logger.info(
            f"Fetching files from FTP date range: {date_value[0]} to {date_value[1]}"
        )

    # Add FTP credentials from environment variables
    ftp_config["ip_address"] = os.getenv("FTP_IP_ADDRESS")
    ftp_config["username"] = os.getenv("FTP_USERNAME")
    ftp_config["password"] = os.getenv("FTP_PASSWORD")

    # Validate FTP credentials
    if not all(
        [ftp_config["ip_address"], ftp_config["username"], ftp_config["password"]]
    ):
        raise RuntimeError(
            "FTP credentials required\n"
            "Set FTP_IP_ADDRESS, FTP_USERNAME, FTP_PASSWORD in .env"
        )

    try:
        from src.ftp_fetcher import FTPFetcher

        fetcher = FTPFetcher(ftp_config)
        downloaded_files = fetcher.fetch_all()
        logger.info(f"✓ FTP fetch complete: {len(downloaded_files)} files downloaded")
        return downloaded_files

    except Exception as e:
        logger.error(f"✗ FTP fetch failed: {e}")
        raise


def process_audio_files(config, input_dir):
    """
    Process audio files using the pipeline.

    Args:
        config (dict): Configuration dictionary
        input_dir (str): Input directory path

    Returns:
        dict: Processing results

    Raises:
        RuntimeError: If any file processing fails
    """
    results = {"total": 0, "successful": 0, "skipped": 0, "failed": 0, "errors": []}

    # Get list of audio files
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return results

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = []

    for extension in audio_extensions:
        audio_files.extend(input_path.glob(f"*{extension}"))

    audio_files.extend(input_path.glob(f"*{extension.upper()}"))

    results["total"] = len(audio_files)

    if results["total"] == 0:
        logger.warning(f"No audio files found in {input_dir}")
        return results

    logger.info(f"Found {results['total']} audio files to process")

    # Initialize pipeline
    try:
        pipeline = Pipeline(config)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

    # Initialize progress tracker
    progress_tracker = ProgressTracker(results["total"])

    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        file_start_time = time.time()

        # Display progress header
        progress_bar, percentage, est_time_str = progress_tracker.get_display_info()
        logger.info(f"[{progress_bar}] | Est. remaining: {est_time_str}")

        try:
            output = pipeline.process_file(str(audio_file))
            file_processing_time = time.time() - file_start_time
            progress_tracker.update(file_processing_time)
            results["successful"] += 1

        except Exception as e:
            file_processing_time = time.time() - file_start_time
            progress_tracker.update(file_processing_time)
            results["failed"] += 1
            error_msg = f"{audio_file.name}: {str(e)}"
            results["errors"].append(error_msg)

            logger.error("✗ Processing stopped due to error")
            logger.error(f"Error: {error_msg}")
            raise RuntimeError(f"Processing failed: {error_msg}")

    return results


def main():
    """Main entry point for Support Call QA System."""
    parser = argparse.ArgumentParser(
        description="Support Call QA System - Process audio files and generate QA assessments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="both",
        choices=["ftp", "local", "both"],
        help="Source of audio files",
    )

    parser.add_argument(
        "--input-dir", type=str, help="Override input directory from config"
    )

    parser.add_argument(
        "--no-gpu-check",
        action="store_true",
        help="Skip GPU availability check (not recommended)",
    )

    parser.add_argument(
        "--ftp-date",
        type=str,
        help="Single date directory (YYYYMMDD format). Not compatible with --ftp-date-start",
    )

    parser.add_argument(
        "--ftp-date-start",
        type=str,
        help="Start date for range (YYYYMMDD format). Requires --ftp-date-end",
    )

    parser.add_argument(
        "--ftp-date-end",
        type=str,
        help="End date for range (YYYYMMDD format). Requires --ftp-date-start",
    )

    parser.add_argument(
        "--list-ftp-dates",
        action="store_true",
        help="List available date directories on FTP server",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear input and output directories before processing",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate support rep report after processing",
    )

    parser.add_argument(
        "--email",
        action="store_true",
        help="Send report via email after processing",
    )

    parser.add_argument(
        "--last-week",
        action="store_true",
        help="Automatically set date range to previous complete week (Mon-Sun)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Handle --email-only mode: skip processing when --email is used without any processing flags
    email_only_mode = args.email and (
        args.source == "both"
        and not args.ftp_date
        and not args.ftp_date_start
        and not args.ftp_date_end
        and not args.last_week
        and not args.report
        and not args.list_ftp_dates
        and not args.clear
    )

    if email_only_mode:
        load_dotenv()
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

        from src.email_sender import send_report_email

        output_dir = config["directories"]["output"]
        email_sent = send_report_email(config, output_dir=output_dir)

        if not email_sent:
            logger.warning("Email could not be sent")
            sys.exit(1)
        sys.exit(0)

    # Handle --last-week flag - auto-calculate previous week (Mon-Sun)
    if args.last_week:
        today = datetime.now()
        current_week_monday = today - timedelta(days=today.weekday())
        last_week_monday = current_week_monday - timedelta(days=7)
        last_week_sunday = last_week_monday + timedelta(days=6)

        args.ftp_date_start = last_week_monday.strftime("%Y%m%d")
        args.ftp_date_end = last_week_sunday.strftime("%Y%m%d")
        logger.info(f"Last week: {args.ftp_date_start} to {args.ftp_date_end}")

    # Load environment variables
    load_dotenv()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("Support Call QA System")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Source: {args.source}")
    logger.info("=" * 70)

    # Check GPU availability
    if not args.no_gpu_check:
        logger.info("Checking GPU availability...")
        if not check_gpu_availability():
            logger.error(
                "GPU check failed. Use --no-gpu-check to skip (not recommended)"
            )
            sys.exit(1)
    else:
        logger.warning("GPU check skipped via --no-gpu-check flag")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Get input directory
    input_dir = args.input_dir or config["directories"]["input"]
    logger.info(f"Input directory: {input_dir}")

    # Handle directory clearing based on --clear flag
    try:
        confirm_directory_clearing(config, args.clear)
    except Exception as e:
        logger.error(f"Failed to clear directories: {e}")
        sys.exit(1)

    # Handle --list-ftp-dates flag
    if args.list_ftp_dates:
        try:
            list_ftp_directories(config)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to list FTP directories: {e}")
            sys.exit(1)

    # Fetch from FTP if requested
    downloaded_files = []
    if args.source in ["ftp", "both"]:
        # Validation: require date(s) when source is ftp
        if args.source == "ftp" and not (
            args.ftp_date or (args.ftp_date_start and args.ftp_date_end)
        ):
            logger.error(
                "Date required when --source is ftp\n"
                "Choose one of:\n"
                "  - Single date: --ftp-date YYYYMMDD\n"
                "  - Date range: --ftp-date-start YYYYMMDD --ftp-date-end YYYYMMDD"
            )
            sys.exit(1)

        try:
            downloaded_files = fetch_from_ftp(
                config, args.ftp_date, args.ftp_date_start, args.ftp_date_end
            )
        except Exception as e:
            logger.error(f"FTP fetch failed: {e}")
            if args.source == "ftp":
                logger.error("Source is 'ftp' only, exiting")
                sys.exit(1)

    # Process downloaded/local files if requested
    processing_results = None
    if args.source in ["ftp", "local", "both"]:
        try:
            processing_results = process_audio_files(config, input_dir)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            sys.exit(1)

    # Print summary
    logger.info("=" * 70)
    logger.info("Processing Summary")
    logger.info("=" * 70)

    if args.source in ["ftp", "both"]:
        logger.info(f"FTP downloads: {len(downloaded_files)} files")

    if processing_results:
        logger.info(f"Total files found: {processing_results['total']}")
        logger.info(f"Successfully processed: {processing_results['successful']}")
        logger.info(f"Skipped (errors): {processing_results['failed']}")

        if processing_results["errors"]:
            logger.info("\nErrors encountered:")
            for error in processing_results["errors"]:
                logger.info(f"  - {error}")

    logger.info("=" * 70)
    logger.info(f"Completed at: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # Generate report if --report flag is set
    if (
        args.report
        and processing_results
        and processing_results.get("successful", 0) > 0
    ):
        from src.report_generator import generate_report

        output_dir = config["directories"]["output"]
        generate_report(output_dir)

    # Send email if --email flag is set
    if args.email:
        from src.email_sender import send_report_email

        output_dir = config["directories"]["output"]
        email_sent = send_report_email(config, output_dir=output_dir)

        if not email_sent:
            logger.warning("Email could not be sent")

    # Exit with appropriate code
    sys.exit(0)


if __name__ == "__main__":
    main()
