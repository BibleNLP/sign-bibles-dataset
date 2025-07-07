import os
import tkinter as tk
from tkinter import ttk
import queue
import pickle
import boto3
from botocore.exceptions import ClientError
import io


class DownloadLog:
    def __init__(self, log_file):
        self.log_file = log_file
        self.completed_files = set()
        self._load()

    def _load(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "rb") as f:
                    self.completed_files = pickle.load(f)
            except:
                self.completed_files = set()

    def _save(self):
        with open(self.log_file, "wb") as f:
            pickle.dump(self.completed_files, f)

    def mark_completed(self, filepath):
        self.completed_files.add(str(filepath))
        self._save()

    def is_completed(self, filepath):
        return str(filepath) in self.completed_files

    def cleanup(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.completed_files = set()


class DownloadProgressWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DBL Download Progress")
        self.root.geometry("700x350")  # Slightly larger window

        # Configure modern styles
        self.style = ttk.Style(self.root)
        self.style.theme_use("default")

        # Configure colors
        bg_color = "#f8f9fa"  # Light gray background
        frame_bg = "#ffffff"  # White frame background
        text_color = "#212529"  # Dark gray text

        # Configure styles
        self.style.configure("TFrame", background=frame_bg)
        self.style.configure("Background.TFrame", background=bg_color)
        self.style.configure("TLabelframe", background=frame_bg)
        self.style.configure(
            "TLabelframe.Label",
            background=frame_bg,
            foreground=text_color,
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure(
            "Status.TLabel",
            background=bg_color,
            foreground=text_color,
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "File.TLabel",
            background=frame_bg,
            foreground=text_color,
            font=("Segoe UI", 9),
        )

        # Configure progress bar style
        self.style.configure(
            "Blue.Horizontal.TProgressbar",
            troughcolor="#e9ecef",
            background="#007bff",
            lightcolor="#007bff",
            darkcolor="#0056b3",
            bordercolor="#e9ecef",
        )

        # Main background frame
        main_frame = ttk.Frame(self.root, style="Background.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Status label with modern font
        self.status_label = ttk.Label(
            main_frame, text="Initializing...", wraplength=660, style="Status.TLabel"
        )
        self.status_label.pack(pady=(0, 15))

        # Overall progress frame
        progress_frame = ttk.LabelFrame(
            main_frame, text="Overall Progress (0/0)", padding=15
        )
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        self.progress_frame = progress_frame

        self.overall_progress = ttk.Progressbar(
            progress_frame,
            style="Blue.Horizontal.TProgressbar",
            mode="determinate",
            length=600,
        )
        self.overall_progress.pack()

        # Current file frame
        file_frame = ttk.LabelFrame(main_frame, text="Current File", padding=15)
        file_frame.pack(fill=tk.X)

        self.file_progress = ttk.Progressbar(
            file_frame,
            style="Blue.Horizontal.TProgressbar",
            mode="determinate",
            length=600,
        )
        self.file_progress.pack()

        self.file_label = ttk.Label(
            file_frame, text="", wraplength=660, style="File.TLabel"
        )
        self.file_label.pack(pady=(10, 0))

        # Message queue for thread-safe updates
        self.queue = queue.Queue()
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.check_queue)

        # Configure window background
        self.root.configure(bg=bg_color)

        # Center the window on screen
        self.center_window()

    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def on_closing(self):
        """Handle window close event"""
        self.running = False
        self.root.quit()
        self.root.destroy()

    def check_queue(self):
        """Process any pending GUI updates from the queue."""
        if not self.running:
            return

        try:
            for _ in range(100):  # Process up to 100 messages per tick
                try:
                    msg = self.queue.get_nowait()
                    msg_type = msg.get("type")

                    if msg_type == "status":
                        self.status_label.config(text=msg["text"])
                    elif msg_type == "overall_progress":
                        progress = msg.get("value", 0)
                        self.overall_progress["value"] = progress
                        if "current_file" in msg and "total_files" in msg:
                            self.progress_frame.config(
                                text=f"Overall Progress ({msg['current_file']}/{msg['total_files']}) - {progress:.1f}%"
                            )
                    elif msg_type == "file_progress":
                        progress = msg.get("value", 0)
                        self.file_progress["value"] = progress
                        text = msg.get("text", "")
                        if text and progress > 0:
                            text = f"{text} - {progress:.1f}%"
                        self.file_label.config(text=text)
                    elif msg_type == "complete":
                        self.running = False
                        self.root.quit()
                        self.root.destroy()
                        return

                    self.queue.task_done()
                except queue.Empty:
                    break
        finally:
            if self.running:
                self.root.update()  # Use update() instead of update_idletasks()
                self.root.after(50, self.check_queue)

    def update_status(self, text):
        """Thread-safe method to update status text."""
        if self.running:
            self.queue.put({"type": "status", "text": text})

    def update_overall_progress(self, value, current_file=None, total_files=None):
        """Thread-safe method to update overall progress."""
        if self.running:
            self.queue.put(
                {
                    "type": "overall_progress",
                    "value": value,
                    "current_file": current_file,
                    "total_files": total_files,
                }
            )

    def update_file_progress(self, value, text=""):
        """Thread-safe method to update file progress."""
        if self.running:
            self.queue.put({"type": "file_progress", "value": value, "text": text})

    def complete(self):
        """Clean up and close the window."""
        if self.running:
            self.queue.put({"type": "complete"})


class S3Storage:
    def __init__(self, bucket_name, folder="unprocessed"):
        self.s3_client = boto3.client("s3")  # Let boto3 handle credential discovery
        self.bucket_name = bucket_name
        self.folder = folder.rstrip("/")

    def _get_s3_key(self, filepath):
        """Convert local filepath to S3 key."""
        # Extract the language code and project name from the path
        parts = str(filepath).split(os.sep)
        if "downloads" in parts:
            # Remove everything up to and including 'downloads'
            parts = parts[parts.index("downloads") + 1 :]
        return f"{self.folder}/{'/'.join(parts)}"

    def upload_file(self, file_data, filepath, progress_window=None):
        """Upload file data to S3."""
        try:
            s3_key = self._get_s3_key(filepath)

            # Create a wrapper class to track upload progress
            class ProgressPercentage:
                def __init__(self, total_bytes):
                    self._total_bytes = total_bytes
                    self._uploaded = 0

                def __call__(self, bytes_amount):
                    self._uploaded += bytes_amount
                    if progress_window and self._total_bytes > 0:
                        percentage = (self._uploaded * 100) / self._total_bytes
                        progress_window.update_file_progress(
                            percentage, f"Uploading to S3: {s3_key}"
                        )

            # Upload the file with progress tracking
            total_bytes = len(file_data)
            progress_tracker = ProgressPercentage(total_bytes)

            # Create a file-like object from the data
            file_obj = io.BytesIO(file_data)

            # Upload with progress callback
            self.s3_client.upload_fileobj(
                file_obj, self.bucket_name, s3_key, Callback=progress_tracker
            )

            # Verify the upload was successful
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                if progress_window:
                    progress_window.update_file_progress(100, f"Uploaded: {s3_key}")
                return True
            except ClientError:
                if progress_window:
                    progress_window.update_status(f"Failed to verify upload: {s3_key}")
                return False

        except ClientError as e:
            if progress_window:
                progress_window.update_status(f"S3 upload failed: {e}")
            return False
        except Exception as e:
            if progress_window:
                progress_window.update_status(f"Unexpected error during upload: {e}")
            return False

    def file_exists(self, filepath):
        """Check if a file exists in S3."""
        try:
            s3_key = self._get_s3_key(filepath)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            raise e


def validate_mp4(filepath):
    """Validates if a file is a proper MP4 file by checking its signature."""
    try:
        with open(filepath, "rb") as f:
            # Read first 8 bytes
            header = f.read(8)
            if len(header) < 8:
                return False

            # Skip first 4 bytes (file size) and check for 'ftyp' signature
            if header[4:8] != b"ftyp":
                return False

            # Read the next few bytes to check for valid MP4 types
            mp4_type = f.read(4)
            valid_types = [b"mp42", b"isom", b"iso2", b"avc1", b"mp41"]
            return any(mp4_type == type_sig for type_sig in valid_types)
    except Exception:
        return False
