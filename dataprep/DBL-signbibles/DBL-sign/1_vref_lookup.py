#!/usr/bin/env python3
import argparse
import json
import re


def load_vref_map(vref_path: str) -> dict[str, int]:
    with open(vref_path, encoding="utf-8") as f:
        return {line.strip(): idx for idx, line in enumerate(f) if line.strip()}


def load_bible_lines(bible_path: str) -> list[str]:
    with open(bible_path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def parse_citation_string(citation: str, vref_map: dict[str, int]) -> list[int]:
    all_indices = []
    current_book = None
    current_chapter = None
    tokens = re.split(r";\s*", citation.strip())

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)\s+(\d+)", token)
        if m:
            book, chapter = m.groups()
            current_book = book
            current_chapter = chapter
            prefix = f"{book} {chapter}:"
            matches = [i for ref, i in vref_map.items() if ref.startswith(prefix)]
            all_indices.extend(matches)
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)", token)
        if m:
            book = m.group(1)
            current_book = book
            matches = [i for ref, i in vref_map.items() if ref.startswith(f"{book} ")]
            all_indices.extend(matches)
            continue

        m = re.fullmatch(
            r"([1-3]?[A-Z]+)?\s*(\d+:\d+)\s*-\s*([1-3]?[A-Z]+)?\s*(\d+:\d+)", token
        )
        if m:
            book1, start, book2, end = m.groups()
            if book1:
                current_book = book1
            book2 = book2 or current_book
            if not (current_book and book2):
                continue
            start_ref = f"{current_book} {start}"
            end_ref = f"{book2} {end}"
            if start_ref in vref_map and end_ref in vref_map:
                i1, i2 = vref_map[start_ref], vref_map[end_ref]
                if i1 <= i2:
                    all_indices.extend(range(i1, i2 + 1))
            continue

        parts = token.split(",")
        for part in parts:
            part = part.strip()
            m = re.fullmatch(r"([1-3]?[A-Z]+)?\s*(\d+):(\d+(?:-\d+)?)", part)
            if m:
                maybe_book, ch, verse_range = m.groups()
                if maybe_book:
                    current_book = maybe_book
                current_chapter = ch
                if not current_book:
                    continue

                if "-" in verse_range:
                    start_v, end_v = map(int, verse_range.split("-"))
                    verse_numbers = range(start_v, end_v + 1)
                else:
                    verse_numbers = [int(verse_range)]

                for v in verse_numbers:
                    ref = f"{current_book} {current_chapter}:{v}"
                    if ref in vref_map:
                        all_indices.append(vref_map[ref])
    return sorted(set(all_indices))


def citation_to_text_and_vrefs(citation:str, vref_map, bible_lines):
        vrefs = parse_citation_string(citation, vref_map)
        
        verses = [bible_lines[i] for i in vrefs if 0 <= i < len(bible_lines)]

        bible_text = "".join(verses)
        return bible_text, vrefs



        

def main():
    parser = argparse.ArgumentParser(
        description="Augment video JSON with eBible verse indices and text."
    )
    parser.add_argument("vref_path", help="Path to vref.txt file")
    parser.add_argument("json_path", help="Path to input JSON file")
    parser.add_argument("output_json", help="Path to write updated JSON")
    parser.add_argument(
        "--bible-path", help="Path to eBible .txt file (one verse per line)"
    )
    args = parser.parse_args()

    vref_map = load_vref_map(args.vref_path)
    bible_lines = load_bible_lines(args.bible_path)

    with open(args.json_path, encoding="utf-8") as f:
        grouped_data = json.load(f)

    updated_count = 0
    for group in grouped_data:
        for video in group.get("videos", []):
            citation = video.get("bible_passage", "").strip()
            if citation:
                indices = parse_citation_string(citation, vref_map)
                video["ebible_vref_indices"] = indices
                video["bible_text"] = [
                    bible_lines[i] for i in indices if 0 <= i < len(bible_lines)
                ]
                updated_count += 1
            else:
                video["ebible_vref_indices"] = []
                video["bible_text"] = []

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated_count} video entries with verse indices and Bible text.")


if __name__ == "__main__":
    main()

# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/vref_lookup.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/metadata/vref.txt video_file_passages_grouped.json video_file_passages_grouped_with_ebible_vrefs.json
# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/vref_lookup.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/metadata/vref.txt video_file_passages_grouped.json video_file_passages_grouped_with_ebible_vrefs_and_text.json --bible-path /opt/home/cleong/data_munging/local_data/eBible/ebible/corpus/eng-engbsb.txt
