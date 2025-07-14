import argparse
from pathlib import Path


def prepend_to_folder_names(base_dir: Path, prefix: str) -> None:
    for entry in base_dir.iterdir():
        if entry.is_dir():
            new_name = prefix + entry.name
            new_path = base_dir / new_name
            if new_path.exists():
                print(f"Skipping {entry.name} -> {new_name} (already exists)")
            else:
                print(f"Renaming {entry.name} -> {new_name}")
                entry.rename(new_path)


def main():
    parser = argparse.ArgumentParser(description="Prepend a string to folder names.")
    parser.add_argument("prefix", help="String to prepend to folder names")
    parser.add_argument("directory", help="Path to directory containing folders", type=Path)
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a valid directory.")
        return

    prepend_to_folder_names(args.directory, args.prefix)


if __name__ == "__main__":
    main()
