import gradio as gr
import pandas as pd
from pathlib import Path
import numpy as np

# --- Configuration ---
directory = Path(".").resolve()
csv_files = sorted(directory.rglob("*.ocr.textchanges.csv"))
current_file_index = 0  # To keep track of the currently loaded file


# --- Helper Functions ---
def load_csv_from_index(index):
    """Loads a CSV file based on its index in the csv_files list."""
    global current_file_index  # Declare current_file_index as global to modify it

    # Ensure choices are always available for the dropdown even if empty initially
    dropdown_choices = [str(f) for f in csv_files]

    if not csv_files:
        # Return empty dataframe, status, and an empty dropdown
        return (
            pd.DataFrame(),
            "No CSV files found!",
            gr.Dropdown(value=None, choices=[]),
        )

    # Clamp index to be within valid range
    if index < 0:
        index = 0
    elif index >= len(csv_files):
        index = len(csv_files) - 1

    current_file_index = index
    file_path = str(csv_files[current_file_index])
    df = pd.read_csv(file_path)

    # Return the DataFrame, status message, and a new Dropdown component with the updated value
    return (
        df,
        f"Loaded: {file_path}",
        gr.Dropdown(value=file_path, choices=dropdown_choices),
    )


def load_csv(file_path):
    """Loads a CSV file when selected from the dropdown."""
    global current_file_index
    dropdown_choices = [str(f) for f in csv_files]  # Ensure choices are always passed

    if file_path is None or not Path(file_path).exists():
        return (
            pd.DataFrame(),
            "Please select a valid file.",
            gr.Dropdown(value=None, choices=dropdown_choices),
        )

    try:
        # Find the index of the selected file to update current_file_index
        index = csv_files.index(Path(file_path))
        current_file_index = index
        df = pd.read_csv(file_path)
        # Return the DataFrame, status message. Dropdown value will already be correct.
        return df, f"Loaded: {file_path}"
    except ValueError:
        # This case should ideally not happen if file_path comes from dropdown choices
        return (
            pd.DataFrame(),
            "File not found in list.",
            gr.Dropdown(value=file_path, choices=dropdown_choices),
        )


def save_csv(edited_df, file_path):
    """Saves the edited DataFrame back to the CSV."""
    if file_path is None:
        return "Error: No file selected to save."

    df = pd.DataFrame(edited_df)
    # Ensure all columns exist before dropping NA, then save
    if not df.empty:
        # Convert any empty strings to NaN so dropna can catch them
        # This loop iterates through columns and replaces empty strings with NaN
        # It's robust for columns that might have different dtypes.
        for col in df.columns:
            if (
                df[col].dtype == "object"
            ):  # Check if column is of object type (likely strings)
                df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
                # The regex r'^\s*$' matches empty strings or strings with only whitespace

        df.dropna(how="all").to_csv(file_path, index=False)
    else:
        # If the DataFrame is empty after editing, create an empty file or handle as needed
        pd.DataFrame().to_csv(file_path, index=False)
    return f"{file_path} Saved!"


def go_next():
    """Moves to the next CSV file in the list."""
    new_index = current_file_index + 1
    return load_csv_from_index(new_index)


def go_previous():
    """Moves to the previous CSV file in the list."""
    new_index = current_file_index - 1
    return load_csv_from_index(new_index)


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# CSV Editor with Navigation")

    # Components
    with gr.Row():
        dropdown = gr.Dropdown(
            choices=[str(f) for f in csv_files], label="Select CSV File", scale=3
        )
        previous_button = gr.Button("Previous", scale=1)
        next_button = gr.Button("Next", scale=1)

    table = gr.Dataframe(label="Edit CSV Content", interactive=True)
    save_button = gr.Button("Save Changes")
    status = gr.Textbox(label="Status", interactive=False)

    # Event Listeners
    # When dropdown changes, load the selected CSV
    dropdown.change(
        fn=load_csv,
        inputs=dropdown,
        outputs=[table, status],  # No need to update dropdown here as it's the input
    )

    # When "Previous" is clicked, go to the previous file
    previous_button.click(
        fn=go_previous,
        inputs=[],
        outputs=[table, status, dropdown],  # Update dropdown with new component
    )

    # When "Next" is clicked, go to the next file
    next_button.click(
        fn=go_next,
        inputs=[],
        outputs=[table, status, dropdown],  # Update dropdown with new component
    )

    # When "Save" is clicked, save the current edited DataFrame
    save_button.click(
        fn=save_csv,
        inputs=[
            table,
            dropdown,
        ],  # Pass the edited table data and the current file path
        outputs=status,
    )

    # Initial load: load the first CSV if any exist
    # The lambda function here provides the initial outputs for the components
    demo.load(
        fn=lambda: load_csv_from_index(0)
        if csv_files
        else (
            pd.DataFrame(),
            "No CSV files found!",
            gr.Dropdown(value=None, choices=[]),
        ),
        outputs=[table, status, dropdown],
    )


demo.launch()
