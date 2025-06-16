#!/usr/bin/env python3
import argparse
import re
from typing import Dict, List

def load_vref_map(vref_path: str) -> Dict[str, int]:
    """Build mapping from verse reference to index from vref.txt."""
    with open(vref_path, encoding='utf-8') as f:
        return {line.strip(): idx for idx, line in enumerate(f) if line.strip()}

def parse_citation_string(citation: str, vref_map: Dict[str, int]) -> List[int]:
    all_indices = []
    current_book = None
    current_chapter = None
    tokens = re.split(r';\s*', citation.strip())

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Match full chapter: "GEN 6"
        m = re.fullmatch(r'([1-3]?[A-Z]+)\s+(\d+)', token)
        if m:
            book, chapter = m.groups()
            current_book = book
            current_chapter = chapter
            prefix = f"{book} {chapter}:"
            matches = [i for ref, i in vref_map.items() if ref.startswith(prefix)]
            all_indices.extend(matches)
            continue

        # Match full book: "MRK"
        m = re.fullmatch(r'([1-3]?[A-Z]+)', token)
        if m:
            book = m.group(1)
            current_book = book
            matches = [i for ref, i in vref_map.items() if ref.startswith(f"{book} ")]
            all_indices.extend(matches)
            continue

        # Match cross-chapter range: "GEN 6:1 - 7:24"
        m = re.fullmatch(
            r'([1-3]?[A-Z]+)?\s*(\d+:\d+)\s*-\s*([1-3]?[A-Z]+)?\s*(\d+:\d+)',
            token
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

        # Match intra-chapter or multi-chapter sets: "GEN 18:17-33,19:1-29"
        parts = token.split(",")
        for part in parts:
            part = part.strip()
            # Match full "BOOK CH:VV-VV"
            m = re.fullmatch(r'([1-3]?[A-Z]+)?\s*(\d+):(\d+(?:-\d+)?)', part)
            if m:
                maybe_book, ch, verse_range = m.groups()
                if maybe_book:
                    current_book = maybe_book
                current_chapter = ch
                if not current_book:
                    continue

                if '-' in verse_range:
                    start_v, end_v = map(int, verse_range.split('-'))
                    verse_numbers = range(start_v, end_v + 1)
                else:
                    verse_numbers = [int(verse_range)]

                for v in verse_numbers:
                    ref = f"{current_book} {current_chapter}:{v}"
                    if ref in vref_map:
                        all_indices.append(vref_map[ref])
            else:
                # Might be malformed or incomplete
                continue

    return sorted(set(all_indices))

def indices_to_verses(indices: list[int], vref_path: str) -> list[str]:
    """Given a list of indices, return the corresponding verse strings from vref.txt."""
    with open(vref_path, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return [lines[i] for i in indices if 0 <= i < len(lines)]



def main():
    parser = argparse.ArgumentParser(description="Convert verse references to index list.")
    parser.add_argument("vref_path", help="Path to vref.txt file")
    parser.add_argument("input_path", help="Path to file with verse reference strings")
    args = parser.parse_args()

    vref_map = load_vref_map(args.vref_path)

    with open(args.input_path, encoding='utf-8') as f:
        for line in f:
            citation = line.strip()
            if citation:
                indices = parse_citation_string(citation, vref_map)
                verses = indices_to_verses(indices, args.vref_path)
                print(f"{citation}\t: {len(indices)} indices, {len(verses)} verses")
                # for index, verse in zip(indices, verses):
                #     print(f"{index}, {verse}")



if __name__ == "__main__":
    main()


# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/vref_lookup.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/metadata/vref.txt /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/bible_passages.txt
