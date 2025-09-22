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

        m = re.fullmatch(r"([1-3]?[A-Z]+)?\s*(\d+:\d+)\s*-\s*([1-3]?[A-Z]+)?\s*(\d+:\d+)", token)
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


# TODO: test "Genesis 7:11,7:17-20; 8:2; 7:22-23"
# TODO: Genesis 6:3,5-7
# "Genesis 7:5-10,13-16"
# "Genesis 7:24; 8:1, 3-4"
# Genesis 5:13;6:1
# "1 Samuel 17:12-15,17-19"
# "Matthew 27:51,54;Luke 23:46;John 19:28,30"
# Exodus 12:37-38,40; 13:19
# "1 Samuel 17:54,57-58"
# Daniel 5:31-6:3
# "Matthew 27:51,54;Luke 23:46;John 19:28,30"
def citation_to_text_and_vrefs(citation: str, vref_map, bible_lines):
    vrefs = parse_citation_string(citation, vref_map)

    verses = [bible_lines[i] for i in vrefs if 0 <= i < len(bible_lines)]

    bible_text = "".join(verses)
    return bible_text, vrefs


def main():
    parser = argparse.ArgumentParser(description="Augment video JSON with eBible verse indices and text.")
    parser.add_argument("vref_path", help="Path to vref.txt file")
    parser.add_argument("json_path", help="Path to input JSON file, or dir of json files")

    parser.add_argument("bible_path", help="Path to eBible .txt file (one verse per line)")
    parser.add_argument("--iso_code", help="ISO 639-3 for the Bible")
    parser.add_argument("--bcp_code", help="BCP-47 for the Bible")
    parser.add_argument("--output_json", help="Path to write updated JSON")
    args = parser.parse_args()
    raise DeprecationWarning("This code is outdatated, look in ebible_utils")

    vref_map = load_vref_map(args.vref_path)
    bible_lines = load_bible_lines(args.bible_path)

    with open(args.json_path, encoding="utf-8") as f:
        video_metadata = json.load(f)
        transcripts = []
        for transcript in video_metadata:
            citation = transcript.get("bible-ref", "").strip()
            if citation:
                indices = parse_citation_string(citation, vref_map)
                transcript["biblenlp-vref"] = indices
                transcript["bible_text"] = "".join([bible_lines[i] for i in indices if 0 <= i < len(bible_lines)])
                transcript["language"] = {
                    "name": "English",
                    "ISO639-3": "eng",
                    "BCP-47": "en-US",
                }
                transcript["source"] = "Berean Standard Bible"
                transcripts.append(transcript)

    if args.output_json:
        output_json = args.output_json
    else:
        output_json = args.json_path
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)
        print(f"Wrote output to {output_json}")


if __name__ == "__main__":
    main()

# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/vref_lookup.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/metadata/vref.txt video_file_passages_grouped.json video_file_passages_grouped_with_ebible_vrefs.json
# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/vref_lookup.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/metadata/vref.txt video_file_passages_grouped.json video_file_passages_grouped_with_ebible_vrefs_and_text.json --bible-path /opt/home/cleong/data_munging/local_data/eBible/ebible/corpus/eng-engbsb.txt
