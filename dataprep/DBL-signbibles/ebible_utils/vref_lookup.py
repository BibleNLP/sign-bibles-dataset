# vref_lookup.py
import re
from typing import List, Dict
import csv


def load_vref_map(vref_path: str) -> Dict[str, int]:
    with open(vref_path, encoding="utf-8") as f:
        return {line.strip(): idx for idx, line in enumerate(f) if line.strip()}


def load_bible_lines(bible_path: str) -> List[str]:
    with open(bible_path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def expand_compound_citations(citation: str) -> str:
    """
    Expand comma-separated verse ranges to include full book/chapter references,
    e.g., "Genesis 6:3,5-7" becomes "Genesis 6:3; Genesis 6:5-7"
    """
    tokens = re.split(r";\s*", citation.strip())
    expanded = []

    current_book = None
    current_chapter = None

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        parts = token.split(",")
        for i, part in enumerate(parts):
            part = part.strip()

            # Book and chapter present: "Genesis 6:3"
            m1 = re.fullmatch(r"([1-3]?\s*[A-Za-z]+)\s+(\d+):(\d+(?:-\d+)?)", part)
            if m1:
                current_book, current_chapter, _ = m1.groups()
                expanded.append(part)
                continue

            # Chapter present: "6:3"
            m2 = re.fullmatch(r"(\d+):(\d+(?:-\d+)?)", part)
            if m2:
                current_chapter, _ = m2.groups()
                expanded.append(f"{current_book} {current_chapter}:{m2.group(2)}")
                continue

            # Just a verse or verse range: "3" or "5-7"
            m3 = re.fullmatch(r"(\d+(?:-\d+)?)", part)
            if m3:
                expanded.append(f"{current_book} {current_chapter}:{m3.group(1)}")
                continue

            # Fully-qualified already? Just add it
            expanded.append(part)

    return "; ".join(expanded)


def parse_citation_string(citation: str, vref_map: Dict[str, int]) -> List[int]:
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
            all_indices.extend(
                i for ref, i in vref_map.items() if ref.startswith(prefix)
            )
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)", token)
        if m:
            book = m.group(1)
            current_book = book
            all_indices.extend(
                i for ref, i in vref_map.items() if ref.startswith(f"{book} ")
            )
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

            # Match optional book and chapter: GEN 6:3 or 6:3 or 3
            m = re.fullmatch(r"([1-3]?[A-Z]+)?\s*(\d+)?(?::)?(\d+(?:-\d+)?)", part)
            if m:
                maybe_book, maybe_ch, verse_range = m.groups()

                # Fallback to last known book/chapter
                book = maybe_book or current_book
                chapter = maybe_ch or current_chapter

                if not book or not chapter:
                    continue  # skip if insufficient context

                current_book = book
                current_chapter = chapter

                if "-" in verse_range:
                    start_v, end_v = map(int, verse_range.split("-"))
                    verse_numbers = range(start_v, end_v + 1)
                else:
                    verse_numbers = [int(verse_range)]

                for v in verse_numbers:
                    ref = f"{book} {chapter}:{v}"
                    if ref in vref_map:
                        all_indices.append(vref_map[ref])

    return sorted(set(all_indices))


def citation_to_text_and_vrefs(
    citation: str,
    vref_map: dict[str, int],
    bible_lines: list[str],
    book_map: dict[str, str],
) -> tuple[str, list[int]]:
    normalized = normalize_book_names(citation, book_map)
    expanded = expand_compound_citations(normalized)
    vrefs = parse_citation_string(expanded, vref_map)
    verses = [bible_lines[i] for i in vrefs if 0 <= i < len(bible_lines)]
    return "".join(verses), vrefs


def load_usfm_book_map(csv_path: str) -> Dict[str, str]:
    """
    Load a mapping from English book names (like 'Genesis') to USFM codes (like 'GEN').
    Also handles books with numeric prefixes like '1 Samuel' -> '1SA'.
    """
    book_map = {}

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row["English Name"].strip()
            code = row["Identifier"].strip()
            book_map[name] = code

            # Also add numeric versions (e.g., "1 Samuel" -> "1SA")
            if name[0].isdigit() and " " in name:
                book_map[name] = code
            elif " " in name:
                # Add numbered books explicitly like "1 Samuel" or "2 Kings"
                for prefix in ["1", "2", "3"]:
                    compound_name = f"{prefix} {name}"
                    compound_code = f"{prefix}{code}"
                    book_map[compound_name] = compound_code

    return book_map


def normalize_book_names(citation: str, book_map: Dict[str, str]) -> str:
    """
    Replace human-readable book names with USFM codes (e.g., 'Genesis' -> 'GEN')
    so that we can match against vref.txt properly.
    """
    # Sort keys by descending length to prevent partial replacements (e.g., "John" before "1 John")
    for book_name in sorted(book_map, key=len, reverse=True):
        usfm_code = book_map[book_name]
        pattern = r"\b" + re.escape(book_name) + r"\b"
        citation = re.sub(pattern, usfm_code, citation)
    return citation
