import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def get_mis_code(first_video_path):
    filename = Path(first_video_path).name
    parts = filename.split("-")
    return parts[2] if len(parts) >= 3 else "mis"


def resolve_lang_code(entry):
    lang_code = entry["language_code"]
    videos = entry.get("videos", [])
    if not videos:
        return None, None

    first_video_path = videos[0]["mp4_path"]
    project_dir_name = Path(first_video_path).parent.name

    for prefix in [
        "Chronological Bible Translation in ",
        "The Bible in ",
    ]:
        if project_dir_name.startswith(prefix):
            simplified = project_dir_name[len(prefix) :]
            break
    else:
        simplified = project_dir_name

    if lang_code == "mis":
        inferred_code = get_mis_code(first_video_path)
        real_code = f"mis-{inferred_code}"
    else:
        real_code = lang_code

    return real_code, simplified


def parse_json(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    summaries = defaultdict(
        lambda: {
            "project_names": set(),
            "video_count": 0,
            "total_verses": 0,
        }
    )

    for entry in data:
        real_code, simplified_project_name = resolve_lang_code(entry)
        if real_code is None:
            continue

        summaries[real_code]["project_names"].add(simplified_project_name)
        summaries[real_code]["video_count"] += len(entry["videos"])
        summaries[real_code]["total_verses"] += sum(len(v.get("ebible_vref_indices", [])) for v in entry["videos"])

    return summaries


def generate_latex_table(rows, total_projects, total_videos, total_verses):
    print(r"""\begin{table*}[t]
\centering
\begin{tabular}{p{4cm}@{\hskip .5cm}p{6cm}@{\hskip .5cm}r@{\hskip .5cm}r@{\hskip .5cm}r}
\toprule
\textbf{Language Code} & \textbf{Projects} & \textbf{\# Projects} & \textbf{\# Videos} & \textbf{\# Verses} \\
\midrule""")

    for row in rows:
        lang_code, projects_str, project_count, video_count, verse_count = row
        projects_str = projects_str.replace("_", r"\_")  # Escape underscores
        lang_code = lang_code.replace("_", r"\_")  # Escape underscores
        print(f"{lang_code} & {projects_str} & {project_count} & {video_count} & {verse_count} \\\\")

    print(r"""\midrule""")
    print(rf"\textbf{{TOTAL}} & -- & {total_projects} & {total_videos} & {total_verses} \\")
    print(r"""\bottomrule
\end{tabular}
\caption{Summary of Deaf Bible video data: language codes, project names, video and verse counts.}
\label{tab:deaf-bible-summary}
\end{table*}""")


def main():
    parser = argparse.ArgumentParser(description="Parse Deaf Bible JSON metadata for summary statistics.")
    parser.add_argument("json_path", type=Path, help="Path to the JSON file to parse")
    parser.add_argument("--csv-output", type=Path, help="Optional path to save CSV output")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table instead of plain text")
    args = parser.parse_args()

    summaries = parse_json(args.json_path)

    total_projects = 0
    total_videos = 0
    total_verses = 0
    output_rows = []

    for lang_code in sorted(summaries):
        summary = summaries[lang_code]
        project_names = sorted(summary["project_names"])
        project_count = len(project_names)
        total_projects += project_count
        total_videos += summary["video_count"]
        total_verses += summary["total_verses"]
        project_str = ", ".join(project_names)

        output_rows.append(
            (
                lang_code,
                project_str,
                project_count,
                summary["video_count"],
                summary["total_verses"],
            )
        )

    # Print LaTeX or plain output
    if args.latex:
        generate_latex_table(output_rows, total_projects, total_videos, total_verses)
    else:
        for row in output_rows:
            print(f"{row[0]}\t{row[1]}\t{row[2]} project(s)\t{row[3]} videos\t{row[4]} verses")
        print("-" * 80)
        print(
            f"TOTAL\t{total_projects} project(s) across {len(summaries)} language codes\t"
            f"{total_videos} videos\t{total_verses} verses"
        )

    # Optional CSV output
    if args.csv_output:
        with args.csv_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Lang Code", "Projects", "# Projects", "# Videos", "# Verses"])
            writer.writerows(output_rows)
            writer.writerow(["TOTAL", "--", total_projects, total_videos, total_verses])


if __name__ == "__main__":
    main()
