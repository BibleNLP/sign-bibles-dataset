import json
import re


bibles = { "ERV": "./ERV_bible.json", "BSB": "./BSB_bible.json", "WEB": "./WEB_bible.json"}

def load_bible(bible_json):
	with open(bible_json, "r", encoding='utf-8') as fp:
		bible_dict = json.load(fp)
	return bible_dict

for bib in bibles:
	path = bibles[bib]
	bibles[bib] = load_bible(path)


ref_pattern = re.compile(r"(\w+) (\d+):(.*)")
verse_pattern1 = re.compile(r"(\d+)$")
verse_pattern_range = re.compile(r"(\d+)-(\d+)")

def get_verses(ref, bible_name="WEB"):
	bible_dict = bibles[bible_name]
	match = re.match(ref_pattern, ref)
	if match:
		book = match.group(1)
		if book in book_codes:
			buk = book
		else:
			buk = book_code_lookup[book]
		chap = match.group(2)
		verses = match.group(3)
		v_match = re.match(verse_pattern1, verses)
		if v_match:
			return bible_dict[f'{buk} {chap}:{verses}']
		v_match2 = re.match(verse_pattern_range, verses)
		if v_match2:
			start = v_match2.group(1)
			end = v_match2.group(2)
			v_list = range(int(start), int(end)+1)
			text = []
			for v in v_list:
				try:
					text.append(bible_dict[f"{buk} {chap}:{v}"])
				except Exception as exce:
					if f"{buk} {chap}:{v}" not in known_missing_verses[bible_name]:
						raise Exception(f"Cannot find: {buk} {chap}:{v}") from exce
			return "\n".join(text)
		raise Exception(f"Cannot process verse part of the input reference:{verses}")
	raise Exception(f"Cannot process input reference pattern:{ref}")


book_code_lookup = {
	"matthew" : "MAT",
	"mark" : "MRK",
	"luke" : "LUK",
	"john" : "JHN"
}

book_codes = ["MAT", "MRK", "LUK", "JHN"]

known_missing_verses = {
	"BSB": [
		"MAT 17:21",
		"MAT 18:11",
		"MAT 23:14",
		"MRK 7:16",
		"MRK 9:44",
		"MRK 9:46",
		"MRK 11:26",
		"MRK 15:28",
		"LUK 17:36",
		"LUK 23:17",
		"JHN 5:4"
	],
	"ERV": [
		"MAT 17:21",
		"MAT 18:11",
		"MAT 23:14",
		"MRK 7:16",
		"MRK 9:44",
		"MRK 9:46",
		"MRK 11:26",
		"MRK 15:28",
		"LUK 17:36",
		"LUK 23:17",
		"JHN 5:4"
	]
}

if __name__ == '__main__':
	print(get_verses("Matthew 1:1-5"))
	print(get_verses("LUK 3:16"))
