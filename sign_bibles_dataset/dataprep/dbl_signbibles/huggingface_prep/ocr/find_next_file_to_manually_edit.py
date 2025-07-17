import argparse
import logging
from pathlib import Path

import pandas as pd

# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def process_first_ocr_textchanges_csv(directory: Path) -> None:
    """
    Searches a specified directory recursively for files matching the pattern
    '*.ocr.textchanges.csv'. It then sorts the found files.
    It iterates through the sorted list to find the first file that does NOT
    already have a corresponding '*.manualedit.csv' file in the same directory.
    Once such a file is found, it loads its data, saves it back to a new
    '*.ocr.manualedit.csv' file, and then prints the absolute path of the newly
    created file. The function will process only one such file per run.

    Args:
        directory: A `pathlib.Path` object representing the root directory
                   to begin the recursive search.

    Raises:
        FileNotFoundError: If the specified directory does not exist or a file
                           expected to exist is not found during processing.
        pd.errors.EmptyDataError: If the selected CSV file is empty.
        pd.errors.ParserError: If pandas encounters an error parsing the CSV file.
        Exception: For any other unexpected errors during the process.

    """
    # Validate that the provided path is indeed an existing directory
    if not directory.is_dir():
        logging.error(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    logging.info(f"Starting search for '*.ocr.textchanges.csv' files in '{directory}'...")
    try:
        # Recursively find all files matching the pattern and sort them alphabetically.
        # This ensures consistent selection order.
        ocr_files = sorted(directory.rglob("*.ocr.textchanges.csv"))
        logging.info(f"Found {len(ocr_files)} files")

        # Check if any files were found at all
        if not ocr_files:
            logging.info(f"No '*.ocr.textchanges.csv' files found in '{directory}'.")
            return

        file_processed = False
        # Iterate through the sorted files to find the first one that doesn't
        # already have a corresponding '.manualedit.csv' file.

        for i, current_file in enumerate(ocr_files):
            # Construct the potential output file path for the current file.
            # Example: 'path/to/my_document.ocr.textchanges.csv'
            # will become 'path/to/my_document.ocr.manualedit.csv'.
            new_stem = current_file.stem.replace(".ocr.textchanges", ".ocr.manualedit")
            output_path = current_file.with_stem(new_stem)

            # Check if the output file already exists
            if not output_path.exists():
                logging.info(f"Selected file: '{current_file.relative_to(directory)}'")
                logging.info(f"Output file '{output_path.relative_to(directory)}' does not exist. Processing...")

                logging.info(f"Loading data from '{current_file.relative_to(directory)}'...")
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(current_file)

                # ✅ Edit 1: Read CSV with specific dtypes
                df = pd.read_csv(current_file, dtype={"frame_index": int, "text": str})

                # ✅ Edit 2: Sort by frame_index ascending
                df = df.sort_values("frame_index", ascending=True)

                # Replace periods with colons in the 'text' column
                df["text"] = df["text"].str.replace(".", ":", regex=False)

                # ✅ Edit 4: Strip trailing 'oooo' or similar patterns (case-insensitive, 3+ 'o's) and anything after
                # e.g. Doooo, oooo
                df["text"] = df["text"].str.replace(r"[Oo]{3,}.*$", "", regex=True).str.strip()

                logging.info(f"Saving data to '{output_path.relative_to(directory)}'...")
                # Save the DataFrame to the new CSV file.
                # `index=False` prevents pandas from writing the DataFrame index as a column.
                df.to_csv(output_path, index=False)

                # Print the full, absolute path of the newly created file
                logging.info(f"Successfully created and saved: \n{output_path.resolve()}")
                logging.info(
                    f"Original file should be at \n{str(output_path.resolve()).replace('.ocr.manualedit.csv', '.mp4')}"
                )
                print(f"FYI that was {i}/{len(ocr_files)}")
                file_processed = True
                break  # Exit the loop after processing the first eligible file
            else:
                logging.info(
                    f"Skipping '{current_file.relative_to(directory)}' as "
                    f"'{output_path.relative_to(directory)}' already exists."
                )

        if not file_processed:
            logging.info(
                "All found '*.ocr.textchanges.csv' files already have "
                "corresponding '*.manualedit.csv' files. No new file was created."
            )

    except FileNotFoundError:
        logging.error(
            "Error: A file was not found during processing. "
            "This might indicate an issue with file access or a race condition."
        )
    except pd.errors.EmptyDataError:
        logging.error(f"Error: The selected file '{current_file.relative_to(directory)}' is empty.")
    except pd.errors.ParserError:
        logging.error(
            f"Error: Could not parse '{current_file.relative_to(directory)}'. Please check if it is a valid CSV file."
        )
    except Exception as e:
        # Catch any other unexpected errors and log them
        logging.error(f"An unexpected error occurred during processing: {e}")


if __name__ == "__main__":
    # Set up argument parsing for command-line execution
    parser = argparse.ArgumentParser(
        description=(
            "Finds the first '*.ocr.textchanges.csv' file in a directory (recursively) "
            "that does not already have a corresponding '*.manualedit.csv' file. "
            "It then copies its content to a new '*.manualedit.csv' file in the same location, "
            "and prints the path of the new file. Only one file is processed per run."
        ),
        formatter_class=argparse.RawTextHelpFormatter,  # Helps with multiline descriptions
    )
    parser.add_argument(
        "directory",
        type=Path,  # Automatically converts the string argument to a Path object
        help="The root directory to search for OCR text changes CSV files.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main processing function with the provided directory
    process_first_ocr_textchanges_csv(args.directory)

# workflow:
# first go through all the ocr files, manually correcting
# conda activate /opt/home/cleong/envs/sign-bibles-dataset && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/ocr/find_next_file_to_manually_edit.py "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/downloads/ase/Chronological Bible Translation in American Sign Language (119 Introductions and Passages)"
# then add thumbnails
# # python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/ocr/extract_thumbnails_from_manualedit.py "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/downloads/ase/Chronological Bible Translation in American Sign Language (119 Introductions and Passages)"
# then preview with
# streamlit run /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/ocr/view_manualedit_frames.py
