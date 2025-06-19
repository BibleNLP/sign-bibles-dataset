import os
import json
import re

out_file = "ERV_bible.json"
in_folder = "../../../engerv_readaloud/"

bible_json = {}
files = os.listdir(in_folder)

file_name_pattern = re.compile(r'engerv_\d\d\d_(\w\w\w)_(\d+)_read\.txt')
files = list(sorted(files))
for filepath in files:
	match = re.match(file_name_pattern, filepath)
	if not match:
		# print(f"Not valid file name: {filepath}")
		continue
		# raise Exception(f"Not valid file name: {filepath}")
	book = match.group(1)
	chapter = int(match.group(2))
	# print(f"{book} {chapter}")
	if chapter == 0:
		continue
	with open(f"{in_folder}/{filepath}", 'r') as fp:
		texts = fp.readlines()[2:]
		for i,line in enumerate(texts):
			line = line.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'").replace('—', '-')
			bible_json[f'{book} {chapter}:{i+1}'] = line[:-1]

with open(out_file, 'w', encoding='utf-8') as ff:
	json.dump(bible_json, ff, indent=2)
