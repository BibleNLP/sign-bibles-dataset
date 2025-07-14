from pathlib import Path
import argparse


def rename_ocr_files_recursive(directory: Path, dry_run: bool = False) -> None:
    """
    Recursively rename files ending in '_frameskip9_minframe0_maxframeNone_ocrtext.json'
    to '.ocr.json'. Supports dry-run and prints a summary.
    """
    suffix_to_replace = "_frameskip9_minframe0_maxframeNone_ocrtext.json"
    new_suffix = ".ocr.json"

    total_files = 0
    matching_files = 0
    renamed_files = 0

    for file_path in directory.rglob("*"):
        if file_path.is_file():
            total_files += 1
            if file_path.name.endswith(suffix_to_replace):
                matching_files += 1
                new_name = file_path.name.removesuffix(suffix_to_replace) + new_suffix
                new_path = file_path.with_name(new_name)

                rel_old = file_path.relative_to(directory)
                rel_new = new_path.relative_to(directory)

                if dry_run:
                    print(f"[DRY RUN] Would rename: {rel_old} -> {rel_new}")
                else:
                    print(f"Renaming: {rel_old} -> {rel_new}")
                    file_path.rename(new_path)
                    renamed_files += 1

    # Summary
    print("\n=== Rename Summary ===")
    print(f"Total files scanned:     {total_files}")
    print(f"Matching files found:    {matching_files}")
    if dry_run:
        print(f"Files that would rename: {matching_files}")
    else:
        print(f"Files renamed:           {renamed_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively rename OCR text files to use .ocr.json suffix."
    )
    parser.add_argument("directory", type=Path, help="Path to the base directory.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print what would be renamed."
    )
    args = parser.parse_args()

    rename_ocr_files_recursive(args.directory, dry_run=args.dry_run)
