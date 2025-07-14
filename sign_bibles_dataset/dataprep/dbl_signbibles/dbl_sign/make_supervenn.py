from pathlib import Path
import argparse
from supervenn import supervenn
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_nonempty_line_indices(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return {i for i, line in enumerate(f) if line.strip()}


def read_sets_from_folder(folder_path: Path):
    sets = []
    labels = []
    line_count = None

    for file_path in tqdm(sorted(folder_path.glob("*"))):
        if not file_path.is_file():
            continue

        nonempty_indices = get_nonempty_line_indices(file_path)
        sets.append(nonempty_indices)
        labels.append(file_path.stem)

        # Check line count consistency
        current_line_count = sum(1 for _ in file_path.open("r", encoding="utf-8"))
        if line_count is None:
            line_count = current_line_count
        elif current_line_count != line_count:
            raise ValueError(
                f"Inconsistent line count in {file_path}: expected {line_count}, got {current_line_count}"
            )

    return sets, labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate Supervenn of nonempty line indices from a folder of files."
    )
    parser.add_argument("folder", type=str, help="Path to folder with input files")
    parser.add_argument(
        "--output", type=str, default="supervenn.png", help="Output image filename"
    )

    args = parser.parse_args()
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder: {folder}")

    sets, labels = read_sets_from_folder(folder)
    print(f"{len(sets)} sets, {len(labels)} labels")

    supervenn(sets, labels, side_plots=True)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Supervenn diagram saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
