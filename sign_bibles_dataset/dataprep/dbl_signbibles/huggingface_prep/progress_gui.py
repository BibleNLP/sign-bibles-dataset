import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
import os
from datetime import datetime


class ProgressGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Bible Processing Progress")
        self.root.geometry("700x500")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Configure the grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        # Create a frame for the header
        header_frame = ttk.Frame(root)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Add title
        title_label = ttk.Label(
            header_frame, text="Sign Bible Video Processing", font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=5)

        # Add current time display
        self.time_label = ttk.Label(header_frame, text="", font=("Arial", 10))
        self.time_label.pack(side=tk.RIGHT, padx=5)

        # Create a frame for overall progress
        progress_frame = ttk.LabelFrame(root, text="Overall Progress")
        progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Overall progress bar
        self.overall_progress_var = tk.DoubleVar(value=0)
        self.overall_progress_label = ttk.Label(progress_frame, text="Overall: 0%")
        self.overall_progress_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.overall_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=600,
            mode="determinate",
            variable=self.overall_progress_var,
        )
        self.overall_progress.grid(row=1, column=0, sticky="ew", padx=5, pady=2)

        # Current operation
        self.current_op_label = ttk.Label(
            progress_frame, text="Current operation: None"
        )
        self.current_op_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)

        # Current operation progress
        self.op_progress_var = tk.DoubleVar(value=0)
        self.op_progress_label = ttk.Label(
            progress_frame, text="Operation progress: 0%"
        )
        self.op_progress_label.grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.op_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=600,
            mode="determinate",
            variable=self.op_progress_var,
        )
        self.op_progress.grid(row=4, column=0, sticky="ew", padx=5, pady=2)

        # Create a frame for the log
        log_frame = ttk.LabelFrame(root, text="Processing Log")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Configure the log frame grid
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        # Add a scrolled text widget for the log
        self.log_text = tk.Text(log_frame, height=15, width=80, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Add a scrollbar to the log text
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        # Add buttons at the bottom
        button_frame = ttk.Frame(root)
        button_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

        # Pause/Resume button
        self.pause_resume_var = tk.StringVar(value="Pause")
        self.pause_resume_button = ttk.Button(
            button_frame,
            textvariable=self.pause_resume_var,
            command=self.toggle_pause_resume,
        )
        self.pause_resume_button.pack(side=tk.LEFT, padx=5)

        # Cancel button
        self.cancel_button = ttk.Button(
            button_frame, text="Cancel", command=self.cancel_processing
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=5)

        # Status variables
        self.paused = False
        self.cancelled = False
        self.running = True

        # Message queue for thread-safe communication
        self.message_queue = queue.Queue()

        # Start the timer update
        self.update_time()

        # Start checking the message queue
        self.check_queue()

    def update_time(self):
        """Update the time display."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def check_queue(self):
        """Check the message queue for updates."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.process_message(message)
                self.message_queue.task_done()
        except queue.Empty:
            pass

        if self.running:
            self.root.after(100, self.check_queue)

    def process_message(self, message):
        """Process a message from the queue."""
        msg_type = message.get("type")

        if msg_type == "log":
            self.add_log(message.get("text"))
        elif msg_type == "overall_progress":
            self.update_overall_progress(message.get("value"))
        elif msg_type == "operation":
            self.update_operation(message.get("text"))
        elif msg_type == "op_progress":
            self.update_op_progress(message.get("value"))
        elif msg_type == "complete":
            self.processing_complete()

    def add_log(self, text):
        """Add text to the log."""
        self.log_text.insert(tk.END, f"{text}\n")
        self.log_text.see(tk.END)

    def update_overall_progress(self, value):
        """Update the overall progress bar."""
        self.overall_progress_var.set(value)
        self.overall_progress_label.config(text=f"Overall: {value:.1f}%")

    def update_operation(self, text):
        """Update the current operation text."""
        self.current_op_label.config(text=f"Current operation: {text}")
        self.op_progress_var.set(0)
        self.op_progress_label.config(text="Operation progress: 0%")

    def update_op_progress(self, value):
        """Update the operation progress bar."""
        self.op_progress_var.set(value)
        self.op_progress_label.config(text=f"Operation progress: {value:.1f}%")

    def processing_complete(self):
        """Handle completion of processing."""
        self.add_log("Processing complete!")
        self.update_overall_progress(100)
        self.update_operation("Complete")
        self.update_op_progress(100)
        self.pause_resume_button.config(state=tk.DISABLED)
        self.cancel_button.config(text="Close")

    def toggle_pause_resume(self):
        """Toggle between pause and resume."""
        self.paused = not self.paused
        if self.paused:
            self.pause_resume_var.set("Resume")
            self.add_log("Processing paused.")
        else:
            self.pause_resume_var.set("Pause")
            self.add_log("Processing resumed.")

    def cancel_processing(self):
        """Cancel the processing."""
        if self.cancel_button.cget("text") == "Close":
            self.on_close()
            return

        self.cancelled = True
        self.add_log("Cancelling processing...")

        # Force exit the application after a short delay
        self.root.after(1000, self._force_exit)

    def _force_exit(self):
        """Force exit the application."""
        self.add_log("Terminating process...")
        # Use os._exit to force terminate the process

        os._exit(0)  # Force terminate the process

    def on_close(self):
        """Handle window close event."""
        self.running = False
        self.cancelled = True
        self.root.destroy()


def start_gui():
    """Start the GUI in a separate thread."""
    root = tk.Tk()
    gui = ProgressGUI(root)
    return root, gui


def update_gui(gui, message):
    """Update the GUI with a message."""
    gui.message_queue.put(message)


def example_usage():
    """Example of how to use the GUI."""
    root, gui = start_gui()

    # Simulate processing in a separate thread
    def process_task():
        # Overall progress steps
        steps = [
            "Downloading videos",
            "Segmenting videos",
            "Extracting pose data",
            "Creating segmentation masks",
            "Preparing dataset",
        ]

        for i, step in enumerate(steps):
            # Update overall progress
            overall_progress = (i / len(steps)) * 100
            update_gui(gui, {"type": "overall_progress", "value": overall_progress})

            # Update current operation
            update_gui(gui, {"type": "operation", "text": step})
            update_gui(gui, {"type": "log", "text": f"Starting {step}..."})

            # Simulate operation progress
            for j in range(0, 101, 5):
                # Check if paused
                while gui.paused and not gui.cancelled:
                    time.sleep(0.1)

                # Check if cancelled
                if gui.cancelled:
                    update_gui(gui, {"type": "log", "text": "Processing cancelled."})
                    return

                # Update operation progress
                update_gui(gui, {"type": "op_progress", "value": j})

                if j % 20 == 0:
                    update_gui(gui, {"type": "log", "text": f"{step} progress: {j}%"})

                time.sleep(0.1)

            update_gui(gui, {"type": "log", "text": f"Completed {step}."})

        # Processing complete
        update_gui(gui, {"type": "overall_progress", "value": 100})
        update_gui(gui, {"type": "complete"})

    # Start the processing thread
    threading.Thread(target=process_task, daemon=True).start()

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    example_usage()
