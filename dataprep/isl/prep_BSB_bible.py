import os
import json
import re
from usfm_grammar import USFMParser, Filter

in_folder = "../../../engbsb_usfm"
out_file = "BSB_bible.json"

bible_json = {}

files = os.listdir(in_folder)
files = list(sorted(files))

for filepath in files:
	if not filepath.endswith(".usfm"):
		continue
	print(f"Working with {filepath}...")
	with open(f"{in_folder}/{filepath}", 'r', encoding='utf-8') as fp:
		content = fp.read()
	parser = USFMParser(content)
	verses = parser.to_list(include_markers=Filter.TEXT)
	for row in verses[1:]:
		key = f"{row[0]} {row[1]}:{row[2]}"
		line = row[3].replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'").replace('—', '-')
		bible_json[key] = line

with open(out_file, 'w', encoding='utf-8') as fm:
	json.dump(bible_json, fm, indent=2)