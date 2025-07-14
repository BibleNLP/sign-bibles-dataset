"""
Processing Logger Module

This module provides logging functionality for the segmentation processing pipeline.
It creates and manages a run_log.txt file that contains detailed error messages and processing information.
"""

import datetime
import os
import traceback
from enum import Enum


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class ProcessingLogger:
    """
    Logger class for handling processing logs and errors during video segmentation.
    """

    def __init__(self, log_file_path="../output/run_log.txt"):
        """
        Initialize the logger with a path to the log file.

        Args:
            log_file_path (str): Path to the log file

        """
        self.log_file_path = log_file_path
        self.log_dir = os.path.dirname(log_file_path)

        # Create the directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def clear_log(self):
        """Clear the log file at the beginning of a new run."""
        try:
            with open(self.log_file_path, "w") as f:
                f.write(f"=== New Processing Run Started at {datetime.datetime.now()} ===\n\n")
        except Exception as e:
            print(f"Failed to clear log file: {e!s}")

    def log(self, message, level=LogLevel.INFO, segment_name=None, exception=None):
        """
        Log a message to the log file.

        Args:
            message (str): The message to log
            level (LogLevel): The severity level of the log
            segment_name (str, optional): The name of the segment being processed
            exception (Exception, optional): Exception object if an error occurred

        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            segment_info = f" [{segment_name}]" if segment_name else ""

            log_entry = f"[{timestamp}] {level.value}{segment_info}: {message}\n"

            # Add exception traceback if provided
            if exception:
                log_entry += f"Exception details:\n{traceback.format_exc()}\n"

            with open(self.log_file_path, "a") as f:
                f.write(log_entry)

            # Also print to console for immediate feedback
            if level == LogLevel.ERROR:
                print(f"ERROR{segment_info}: {message}")
            elif level == LogLevel.WARNING:
                print(f"WARNING{segment_info}: {message}")
        except Exception as e:
            print(f"Failed to write to log file: {e!s}")

    def log_info(self, message, segment_name=None):
        """Log an informational message."""
        self.log(message, LogLevel.INFO, segment_name)

    def log_warning(self, message, segment_name=None):
        """Log a warning message."""
        self.log(message, LogLevel.WARNING, segment_name)

    def log_error(self, message, segment_name=None, exception=None):
        """Log an error message with optional exception details."""
        self.log(message, LogLevel.ERROR, segment_name, exception)

    def log_debug(self, message, segment_name=None):
        """Log a debug message."""
        self.log(message, LogLevel.DEBUG, segment_name)

    def log_segmentation_error(self, segment_name, method, error_msg, exception=None):
        """
        Log a specific segmentation error.

        Args:
            segment_name (str): Name of the segment being processed
            method (str): The segmentation method that failed
            error_msg (str): Description of the error
            exception (Exception, optional): The exception that was raised

        """
        message = f"Segmentation failed using {method}: {error_msg}"
        self.log_error(message, segment_name, exception)

    def log_ffmpeg_error(self, segment_name, error_msg, exception=None):
        """
        Log a specific ffmpeg-related error during segmentation.

        Args:
            segment_name (str): Name of the segment being processed
            error_msg (str): Description of the error
            exception (Exception, optional): The exception that was raised

        """
        message = f"FFmpeg processing error: {error_msg}"
        self.log_error(message, segment_name, exception)

    def log_segmentation_success(self, segment_name, method, output_path):
        """
        Log successful segmentation.

        Args:
            segment_name (str): Name of the segment being processed
            method (str): The segmentation method used
            output_path (str): Path to the output segmentation file

        """
        message = f"Successfully created segmentation using {method}: {output_path}"
        self.log_info(message, segment_name)


# Create a default logger instance
logger = ProcessingLogger()
