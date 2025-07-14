from pathlib import Path

import gradio as gr
import numpy as np  # Import numpy for np.nan
import pandas as pd

# --- Configuration ---
directory = Path("/data/petabyte/cleong/data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads/ase").resolve()
csv_files = sorted(directory.rglob("*.ocr.textchanges.csv"))
current_file_index = 0  # To keep track of the currently loaded file


# --- Helper Functions ---


def get_manual_edit_path(original_textchanges_path: Path) -> Path:
    """Derives the corresponding .ocr.manualedit.csv path."""
    return original_textchanges_path.parent / original_textchanges_path.name.replace(
        ".ocr.textchanges.csv", ".ocr.manualedit.csv"
    )


def load_csv_from_index(index):
    """
    Loads a CSV file based on its index in the csv_files (textchanges) list.
    Prioritizes .ocr.manualedit.csv; falls back to .ocr.textchanges.csv.
    Also handles bounds checking for the index.
    """
    global current_file_index
    dropdown_choices = [str(f) for f in csv_files]

    if not csv_files:
        return (
            pd.DataFrame(),
            "No CSV files found!",
            gr.Dropdown(value=None, choices=[]),
        )

    # Ensure index is within valid bounds
    if index < 0:
        index = 0
    elif index >= len(csv_files):
        index = len(csv_files) - 1

    current_file_index = index

    # This is the original .ocr.textchanges.csv path from our list
    original_textchanges_path = csv_files[current_file_index]
    # Derive the potential .ocr.manualedit.csv path
    manual_edit_path = get_manual_edit_path(original_textchanges_path)

    df = pd.DataFrame()
    loaded_source_path = ""
    status_message = ""

    # --- Loading Logic (Prioritize manualedit, fallback to textchanges) ---
    if manual_edit_path.exists():
        try:
            df = pd.read_csv(manual_edit_path)
            loaded_source_path = str(manual_edit_path)
            status_message = f"Loaded #{current_file_index + 1}/{len(csv_files)}: {manual_edit_path} (manual edit)"
        except Exception as e:
            status_message = f"Error loading manual edit {manual_edit_path.name}: {e}. Falling back to original..."
            # If manual edit exists but is corrupt, try original
            try:
                df = pd.read_csv(original_textchanges_path)
                loaded_source_path = str(original_textchanges_path)
                status_message += f"\nLoaded {original_textchanges_path} (original)"
            except Exception as e_orig:
                status_message += f"\nError loading original {original_textchanges_path.name}: {e_orig}"
                return (
                    pd.DataFrame(),
                    status_message,
                    gr.Dropdown(value=None, choices=dropdown_choices),
                )
    else:
        # If manual edit does not exist, load the original textchanges file
        try:
            df = pd.read_csv(original_textchanges_path)
            loaded_source_path = str(original_textchanges_path)
            status_message = f"Loaded #{current_file_index + 1}/{len(csv_files)}: {original_textchanges_path} (original, will save to manual edit)"
        except Exception as e:
            status_message = f"Error loading {original_textchanges_path.name}: {e}"
            return (
                pd.DataFrame(),
                status_message,
                gr.Dropdown(value=None, choices=dropdown_choices),
            )

    # Apply the replace operation to the 'text' column if it exists
    if "text" in df.columns:
        df["text"] = df["text"].str.replace(".", ":", regex=False)

    # The dropdown value MUST be the original_textchanges_path because that's what `csv_files` contains
    return (
        df,
        status_message,
        gr.Dropdown(value=str(original_textchanges_path), choices=dropdown_choices),
    )


def load_csv_from_dropdown(file_path_str):
    """
    Loads a CSV file when selected from the dropdown.
    file_path_str will be a .ocr.textchanges.csv path.
    """
    global current_file_index

    selected_path = Path(file_path_str)  # The dropdown passes the string representation

    if not selected_path.exists():  # Basic check if the original textchanges file actually exists
        return (
            pd.DataFrame(),
            "Selected file not found.",
            gr.Dropdown(value=file_path_str, choices=[str(f) for f in csv_files]),
        )

    try:
        # Find the index of the selected file to update current_file_index
        index = csv_files.index(selected_path)
        # Use load_csv_from_index to handle the actual loading priority
        return load_csv_from_index(index)
    except ValueError:
        return (
            pd.DataFrame(),
            "File not found in list.",
            gr.Dropdown(value=file_path_str, choices=[str(f) for f in csv_files]),
        )
    except Exception as e:
        return (
            pd.DataFrame(),
            f"Error selecting file: {e}",
            gr.Dropdown(value=file_path_str, choices=[str(f) for f in csv_files]),
        )


def save_csv(edited_df_list, file_path_str):
    """
    Saves the edited DataFrame back to the CSV.
    Always saves to the .ocr.manualedit.csv file.
    file_path_str is the .ocr.textchanges.csv path of the currently loaded file.
    """
    if file_path_str is None:
        return "Error: No file selected to save."

    df = pd.DataFrame(edited_df_list)

    # Convert any empty strings to NaN so dropna can catch them
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    # Drop rows where ALL values are NaN (including those that were empty strings)
    df = df.dropna(how="all")

    # Get the target manual edit path for saving
    original_textchanges_path = Path(file_path_str)
    target_manual_edit_path = get_manual_edit_path(original_textchanges_path)

    try:
        if not df.empty:
            df.to_csv(target_manual_edit_path, index=False)
        else:
            # If the DataFrame is empty, ensure the file is created as an empty CSV
            pd.DataFrame().to_csv(target_manual_edit_path, index=False)
        return f"{target_manual_edit_path} Saved!"
    except Exception as e:
        return f"Error saving to {target_manual_edit_path.name}: {e}"


def go_next():
    """Moves to the next CSV file in the list."""
    new_index = current_file_index + 1
    return load_csv_from_index(new_index)


def go_previous():
    """Moves to the previous CSV file in the list."""
    new_index = current_file_index - 1
    return load_csv_from_index(new_index)


def go_to_specific_index(index_input_str):
    """Goes to a CSV file at a user-specified index."""
    try:
        target_index = int(index_input_str)
        target_index -= 1  # Adjust for 0-based indexing
        return load_csv_from_index(target_index)
    except ValueError:
        current_file_path_str = str(csv_files[current_file_index]) if csv_files else None
        dropdown_choices = [str(f) for f in csv_files]
        return (
            pd.DataFrame(),
            "Invalid index! Please enter a number.",
            gr.Dropdown(value=current_file_path_str, choices=dropdown_choices),
        )


def reset_current_file_to_original():
    """
    Reloads the current file from its original .ocr.textchanges.csv version,
    discarding all manual edits and ignoring any .ocr.manualedit.csv file.
    """
    global current_file_index

    if not csv_files:
        return (
            pd.DataFrame(),
            "No CSV files to reset!",
            gr.Dropdown(value=None, choices=[]),
        )

    original_textchanges_path = csv_files[current_file_index]
    dropdown_choices = [str(f) for f in csv_files]

    try:
        df = pd.read_csv(original_textchanges_path)
        if "text" in df.columns:
            df["text"] = df["text"].str.replace(".", ":", regex=False)
        status_message = f"Reloaded original copy for: {original_textchanges_path}"
        return (
            df,
            status_message,
            gr.Dropdown(value=str(original_textchanges_path), choices=dropdown_choices),
        )
    except Exception as e:
        status_message = f"Error reloading original {original_textchanges_path.name}: {e}"
        return (
            pd.DataFrame(),  # Return empty df on severe error
            status_message,
            gr.Dropdown(value=str(original_textchanges_path), choices=dropdown_choices),
        )


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# CSV Editor with Manual Edit Priority")

    # File Navigation Row
    with gr.Row():
        dropdown = gr.Dropdown(
            choices=[str(f) for f in csv_files],  # Choices remain the .ocr.textchanges.csv paths
            label="Select CSV File",
            scale=3,
        )
        previous_button = gr.Button("Previous", scale=1)
        next_button = gr.Button("Next", scale=1)

    # New row for "Go to Index"
    with gr.Row():
        index_input = gr.Textbox(
            label="Go to File Index (1-based)",
            placeholder=f"Enter a number between 1 and {len(csv_files)}" if csv_files else "No files found",
        )
        go_to_index_button = gr.Button("Go to Index")

    # Main Dataframe and Buttons
    table = gr.Dataframe(label="Edit CSV Content", interactive=True)
    with gr.Row():
        save_button = gr.Button("Save Changes")
        # Only the "Reset Current File" button remains
        reset_current_button = gr.Button("Reset Current File (Reload Original Copy)")
    status = gr.Textbox(label="Status", interactive=False)

    # --- Event Listeners ---

    dropdown.change(fn=load_csv_from_dropdown, inputs=dropdown, outputs=[table, status, dropdown])

    previous_button.click(
        fn=go_previous,
        inputs=[],
        outputs=[table, status, dropdown],
    )

    next_button.click(
        fn=go_next,
        inputs=[],
        outputs=[table, status, dropdown],
    )

    go_to_index_button.click(
        fn=go_to_specific_index,
        inputs=index_input,
        outputs=[table, status, dropdown],
    )

    save_button.click(fn=save_csv, inputs=[table, dropdown], outputs=status)

    # Connect the "Reset Current File" button
    reset_current_button.click(
        fn=reset_current_file_to_original,
        inputs=[],
        outputs=[table, status, dropdown],
    )

    # Initial load: load the first CSV using the main loading logic
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
