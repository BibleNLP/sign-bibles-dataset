import json
import re


bible_json = "./WEB_bible.json"

with open(bible_json, "r", encoding="utf-8") as fp:
    bible_dict = json.load(fp)

ref_pattern = re.compile(r"(\w+) (\d+):(.*)")
verse_pattern1 = re.compile(r"(\d+)$")
verse_pattern_range = re.compile(r"(\d+)-(\d+)")


def get_verses(ref):
    match = re.match(ref_pattern, ref)
    if match:
        book = match.group(1)
        buk = book_code_lookup[book]
        chap = match.group(2)
        verses = match.group(3)
        v_match = re.match(verse_pattern1, verses)
        if v_match:
            return bible_dict[f"{buk} {chap}:{verses}"]
        v_match2 = re.match(verse_pattern_range, verses)
        if v_match2:
            start = v_match2.group(1)
            end = v_match2.group(2)
            v_list = range(int(start), int(end))
            text = []
            for v in v_list:
                text.append(bible_dict[f"{buk} {chap}:{v}"])
            return "\n".join(text)
        raise Exception(f"Cannot process verse part of the input reference:{verses}")
    raise Exception(f"Cannot process input reference pattern:{ref}")


book_code_lookup = {"Matthew": "MAT", "Mark": "MRK", "Luke": "LUK", "John": "JHN"}

if __name__ == "__main__":
    print(get_verses("Matthew 1:1-5"))
