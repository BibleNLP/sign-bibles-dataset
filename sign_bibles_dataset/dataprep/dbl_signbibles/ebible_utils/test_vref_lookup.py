from pathlib import Path

import pytest
from vref_lookup import (
    citation_to_text_and_vrefs,
    load_bible_lines,
    load_usfm_book_map,
    load_vref_map,
    normalize_book_names,
)


@pytest.fixture(scope="session")
def book_map():
    csv_path = Path(__file__).parent / "data" / "usfm_book_identifiers.csv"
    assert csv_path.exists(), f"Missing: {csv_path}"
    return load_usfm_book_map(str(csv_path))


@pytest.mark.parametrize(
    "original,expected",
    [
        ("Genesis 1:1", "GEN 1:1"),
        ("Exodus 2:3", "EXO 2:3"),
        ("1 Samuel 17:1", "1SA 17:1"),
        ("2 Kings 4:12", "2KI 4:12"),
        ("John 3:16", "JHN 3:16"),
        ("Matthew 1:1; Luke 2:1; John 3:16", "MAT 1:1; LUK 2:1; JHN 3:16"),
        ("Genesis 7:5-10,13-16", "GEN 7:5-10,13-16"),
        ("Daniel 1:1", "DAN 1:1"),
    ],
)
def test_normalize_book_names(original, expected, book_map):
    normalized = normalize_book_names(original, book_map)
    assert normalized == expected


@pytest.fixture(scope="session")
def resources():
    base_dir = Path(__file__).parent / "data"
    vref_path = base_dir / "vref.txt"
    bible_path = base_dir / "eng-engbsb.txt"  # eBible-style, 1 verse per line, order matching vref.txt
    usfm_csv_path = base_dir / "usfm_book_identifiers.csv"

    assert vref_path.exists(), f"Missing: {vref_path}"
    assert bible_path.exists(), f"Missing: {bible_path}"
    assert usfm_csv_path.exists(), f"Missing: {usfm_csv_path}"

    return {
        "vref_map": load_vref_map(str(vref_path)),
        "bible_lines": load_bible_lines(str(bible_path)),
        "book_map": load_usfm_book_map(str(usfm_csv_path)),
    }


@pytest.mark.parametrize(
    "citation,expected_refs",
    [
        (
            "Genesis 1:1",
            ["GEN 1:1"],
        ),
        (
            "Genesis 7:11,7:17-20; 8:2; 7:22-23",
            [
                "GEN 7:11",
                "GEN 7:17",
                "GEN 7:18",
                "GEN 7:19",
                "GEN 7:20",
                "GEN 8:2",
                "GEN 7:22",
                "GEN 7:23",
            ],
        ),
        ("Genesis 6:3,5-7", ["GEN 6:3", "GEN 6:5", "GEN 6:6", "GEN 6:7"]),
        (
            "Genesis 7:5-10,13-16",
            [
                "GEN 7:5",
                "GEN 7:6",
                "GEN 7:7",
                "GEN 7:8",
                "GEN 7:9",
                "GEN 7:10",
                "GEN 7:13",
                "GEN 7:14",
                "GEN 7:15",
                "GEN 7:16",
            ],
        ),
        ("Genesis 7:24; 8:1, 3-4", ["GEN 7:24", "GEN 8:1", "GEN 8:3", "GEN 8:4"]),
        ("Genesis 5:13;6:1", ["GEN 5:13", "GEN 6:1"]),
        (
            "1 Samuel 17:12-15,17-19",
            [
                "1SA 17:12",
                "1SA 17:13",
                "1SA 17:14",
                "1SA 17:15",
                "1SA 17:17",
                "1SA 17:18",
                "1SA 17:19",
            ],
        ),
        (
            "Matthew 27:51,54;Luke 23:46;John 19:28,30",
            ["MAT 27:51", "MAT 27:54", "LUK 23:46", "JHN 19:28", "JHN 19:30"],
        ),
        (
            "Exodus 12:37-38,40; 13:19",
            ["EXO 12:37", "EXO 12:38", "EXO 12:40", "EXO 13:19"],
        ),
        ("1 Samuel 17:54,57-58", ["1SA 17:54", "1SA 17:57", "1SA 17:58"]),
        (
            "Daniel 5:31-6:3",
            ["DAN 5:31", "DAN 6:2", "DAN 6:3"],
        ),  # special case, versification is weird
    ],
)
def test_citation_to_text_and_vrefs(citation, expected_refs, resources):
    vref_map = resources["vref_map"]
    bible_lines = resources["bible_lines"]
    book_map = resources["book_map"]

    bible_text, matched_indices = citation_to_text_and_vrefs(
        citation=citation,
        vref_map=vref_map,
        bible_lines=bible_lines,
        book_map=book_map,
    )

    missing_refs = [ref for ref in expected_refs if ref not in vref_map]
    if missing_refs:
        pytest.skip(f"Skipping test because these refs are missing from vref_map: {missing_refs}")

    expected_indices = sorted(vref_map[ref] for ref in expected_refs)
    assert matched_indices == expected_indices, (
        f"citation: {citation}, \n expected {expected_refs}, \n with indices {expected_indices} \n got {matched_indices}"
    )
    assert all(0 <= idx < len(bible_lines) for idx in matched_indices)
    assert bible_text  # should be non-empty
