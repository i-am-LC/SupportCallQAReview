from ftplib import FTP, error_perm
from datetime import datetime
import os
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FTPFetcher:
    def __init__(self, config):
        self.config = config
        self.ip_address = config.get("ip_address")
        self.username = config.get("username")
        self.password = config.get("password")
        self.download_directory = config.get("download_directory", "input/")

        # REQUIRED: Either single date OR date range
        self.ftp_date = config.get("date")
        self.date_start = config.get("date_start")
        self.date_end = config.get("date_end")

        # Validate that at least one mode is provided
        if not self.ftp_date and not (self.date_start and self.date_end):
            raise RuntimeError(
                "FTP date specification required\n"
                "Use either:\n"
                "  - Single date: --ftp-date YYYYMMDD (CLI) or ftp.date (config)\n"
                "  - Date range: --ftp-date-start YYYYMMDD --ftp-date-end YYYYMMDD (CLI) or ftp.date_start/end (config)"
            )

        # Validate that not both single and range are provided
        if self.ftp_date and (self.date_start or self.date_end):
            raise RuntimeError(
                "Cannot specify both single date and date range\n"
                "Choose either:\n"
                "  - Single date mode\n"
                "  - Date range mode"
            )

        # Validate based on mode
        if self.ftp_date:
            # Single date mode
            self._validate_date_format(self.ftp_date)
        else:
            # Date range mode
            self._validate_date_format(self.date_start)
            self._validate_date_format(self.date_end)

            # Validate start <= end
            start_dt = datetime.strptime(self.date_start, "%Y%m%d")
            end_dt = datetime.strptime(self.date_end, "%Y%m%d")

            if start_dt > end_dt:
                raise RuntimeError(
                    f"Start date ({self.date_start}) must be before or equal to end date ({self.date_end})"
                )

        # Ensure download directory exists
        os.makedirs(self.download_directory, exist_ok=True)

    def is_range_mode(self):
        """
        Check if FTP fetcher is in date range mode.

        Returns:
            bool: True if in range mode, False if in single date mode
        """
        return not self.ftp_date and (self.date_start and self.date_end)

    def _validate_date_format(self, date_str):
        """
        Validate date format is exactly YYYYMMDD.

        Args:
            date_str (str): Date string to validate

        Raises:
            RuntimeError: If date format is invalid
        """
        if not isinstance(date_str, str):
            raise RuntimeError(f"FTP date must be string, got {type(date_str)}")

        # Check format is exactly 8 digits
        if not re.match(r"^\d{8}$", date_str):
            raise RuntimeError(
                f"Invalid FTP date format: {date_str}\n"
                f"Expected format: YYYYMMDD (exactly 8 digits)"
            )

        # Validate it's a real calendar date
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            raise RuntimeError(
                f"Invalid FTP date: {date_str}\n"
                f"Must be a valid calendar date in YYYYMMDD format"
            )

    def _filter_directories_by_date_range(self, directories, start_date, end_date):
        """
        Filter directories by date range (inclusive of both start and end dates).

        Args:
            directories (list): List of directory names
            start_date (str): Start date (YYYYMMDD)
            end_date (str): End date (YYYYMMDD)

        Returns:
            list: Filtered list of directories within date range
        """
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        filtered = []
        for directory in directories:
            try:
                dir_date = datetime.strptime(directory, "%Y%m%d")
                if start_dt <= dir_date <= end_dt:
                    filtered.append(directory)
            except ValueError:
                # Skip directories that don't match YYYYMMDD format
                continue

        return sorted(filtered)  # Sort in chronological order

    def fetch_all(self, file_pattern="*.wav"):
        """
        Fetch files from date range of directories (or single directory).

        For date range mode: Downloads from all directories within the specified
        date range (inclusive of both start and end dates).

        For single date mode: Downloads from the single specified directory.

        Returns:
            list: List of downloaded file paths

        Raises:
            RuntimeError: If date directory doesn't exist (single mode) or
                          no directories in range (range mode)
        """
        downloaded_files = []

        try:
            ftp = self._connect_to_ftp()
            if ftp is None:
                raise RuntimeError("Failed to connect to FTP server")

            # Get all directories
            directories = self._get_directories(ftp)
            logger.info(f"Found {len(directories)} directories on FTP server")

            if self.is_range_mode():
                # Date range mode: filter directories by range
                filtered_dirs = self._filter_directories_by_date_range(
                    directories, self.date_start, self.date_end
                )

                if not filtered_dirs:
                    raise RuntimeError(
                        f"No directories found in date range {self.date_start} to {self.date_end}\n"
                        f"Available directories: {', '.join(directories)}"
                    )

                logger.info(
                    f"Processing {len(filtered_dirs)} directories in date range "
                    f"{self.date_start} to {self.date_end}"
                )

                for directory in filtered_dirs:
                    try:
                        files = self._get_files_in_directory(ftp, directory)
                        logger.info(
                            f"Processing directory: {directory} ({len(files)} files)"
                        )

                        for file in files:
                            if self._is_audio_file(file):
                                downloaded_file = self._download_file(
                                    ftp, directory, file
                                )
                                if downloaded_file:
                                    downloaded_files.append(downloaded_file)

                    except Exception as e:
                        logger.error(f"Error processing directory {directory}: {e}")
                        raise

            else:
                # Single date mode: validate and process single directory
                if self.ftp_date not in directories:
                    raise RuntimeError(
                        f"Date directory not found: {self.ftp_date}\n"
                        f"Available directories: {', '.join(directories)}"
                    )

                # Process ONLY the specified date directory
                logger.info(f"Processing date directory: {self.ftp_date}")

                try:
                    files = self._get_files_in_directory(ftp, self.ftp_date)
                    logger.info(f"Found {len(files)} files in date directory")

                    for file in files:
                        if self._is_audio_file(file):
                            downloaded_file = self._download_file(
                                ftp, self.ftp_date, file
                            )
                            if downloaded_file:
                                downloaded_files.append(downloaded_file)

                except Exception as e:
                    logger.error(f"Error processing directory {self.ftp_date}: {e}")
                    raise

            # Close FTP connection
            ftp.quit()
            logger.info(f"FTP connection closed")

        except Exception as e:
            logger.error(f"Error in FTP fetch: {e}")
            raise

        logger.info(f"FTP fetch complete: {len(downloaded_files)} files downloaded")
        return downloaded_files

    def _connect_to_ftp(self):
        """Establish connection to FTP server."""
        try:
            if not self.ip_address or not self.username or not self.password:
                logger.error("FTP credentials not provided")
                return None

            ftp = FTP(self.ip_address)
            ftp.login(user=self.username, passwd=self.password)
            logger.info(f"Connected to FTP server: {self.ip_address}")
            return ftp

        except Exception as e:
            logger.error(f"Failed to connect to FTP server: {e}")
            return None

    def _get_directories(self, ftp):
        """Get list of directories on FTP server."""
        try:
            directories = []
            for item in ftp.nlst():
                try:
                    # Check if item is a directory
                    current = ftp.pwd()
                    if item not in [".", ".."]:
                        ftp.cwd(item)
                        directories.append(item)
                        ftp.cwd("..")
                except error_perm:
                    # Not a directory, skip
                    pass

            return directories

        except Exception as e:
            logger.error(f"Failed to get directories: {e}")
            return []

    def _get_files_in_directory(self, ftp, directory):
        """Get list of files in a directory."""
        try:
            ftp.cwd(directory)
            files = ftp.nlst()
            ftp.cwd("..")
            return files

        except Exception as e:
            logger.error(f"Failed to get files in directory {directory}: {e}")
            return []

    def _is_audio_file(self, filename):
        """Check if file is an audio file."""
        audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in audio_extensions)

    def _download_file(self, ftp, directory, file):
        """Download file from FTP server."""
        filepath = None
        try:
            ftp.cwd(directory)

            # Get file size for logging
            try:
                file_size = ftp.size(file)
                logger.info(f"  Downloading {file} ({file_size} bytes)")
            except:
                logger.info(f"  Downloading {file}")

            # Construct local filepath
            local_filename = os.path.basename(file)
            filepath = os.path.join(self.download_directory, local_filename)

            # Check if file already exists
            if os.path.exists(filepath):
                logger.warning(f"  File already exists, skipping: {filepath}")
                ftp.cwd("..")
                return None

            # Download file
            with open(filepath, "wb") as output_file:
                ftp.retrbinary(f"RETR {file}", output_file.write)

            logger.info(f"  ✓ Downloaded: {filepath}")
            ftp.cwd("..")
            return filepath

        except Exception as e:
            logger.error(f"  ✗ Failed to download {file}: {e}")
            try:
                ftp.cwd("..")
            except:
                pass
            return None
