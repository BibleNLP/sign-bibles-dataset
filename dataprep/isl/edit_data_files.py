import json
import logging
import argparse
from pathlib import Path
import numpy as np

logging.basicConfig(filename='logs/app.log', level=logging.INFO,
					format='%(asctime)s - %(levelname)s - %(message)s')

DWPOSE_LANDMARKS_NUM = 134

def json_correction(filename):
	orig_data = {}
	with open(filename, 'r', encoding='utf-8') as fp:
		orig_data = json.load(fp)
	orig_data[0]['language']['nameLocal'] = "English"
	with open(filename, 'w', encoding='utf-8') as fp:
		json.dump(orig_data, fp, indent=4)

def npz_correction(filename):
	old_obj = np.load(filename, allow_pickle=True)
	new_array = []
	for item in old_obj['frames']:
		if len(item) < 1:
			item = [np.full(DWPOSE_LANDMARKS_NUM, np.nan, dtype=np.float64)]
		elif len(item > 1):
			item = [item[0]]
		assert len(item) == 1, f"{len(item)=}"
		assert len(item[0]) == DWPOSE_LANDMARKS_NUM, f"{len(item[0])=}"
		new_array.append(item)
	np.savez_compressed(filename, frames=np.array(new_array, dtype=np.float64))

def npz_test(filename):
	new_obj = np.load(filename) # load w/o pickle
	assert 'frames' in new_obj
	assert len(new_obj['frames'][0][0]) == DWPOSE_LANDMARKS_NUM
	assert len(new_obj['frames'][0][0][0]) == 2

def get_known_missing_verses():
	from bible_text_access import known_missing_verses
	from biblenlp_util import ref2vref
	vrefs = set()
	for bible in known_missing_verses:
		for ref in known_missing_verses[bible]:
			vref = ref2vref(ref)
			vrefs.add(vref[0])
	print(f"Known missing verses: {vrefs}")
	return vrefs

def is_missing_versetext(filename, missing_verses):
	json_data = {}
	with open(filename, 'r', encoding='utf-8') as fp:
		json_data = json.load(fp)
	for entry in json_data:
		verses = entry.get('biblenlp-vref', [])
		for verse in verses:
			print(f"Checking {filename} verse {verse}")
			if verse in missing_verses:
				print(f"{filename} has known missing verse text: {verse}")
				return True
	return False


def main():
	parser = argparse.ArgumentParser(description='Calculate total duration from JSON files.')
	parser.add_argument('directories', nargs='+', help='List of directories to search')
	args = parser.parse_args()

	candidate_files = [
		file.with_suffix('.json')
		for directory in args.directories 
			# for file in Path(directory).rglob("*.transcripts.json")
			for file in Path(directory).rglob("*.mp4") 
		# for file in Path(directory).rglob("*.npz")
	]

	missing_verses = get_known_missing_verses()
	print(f"Total candidate files: {len(candidate_files)}")
	print(f"Candidate files: {candidate_files[:10]} ...")

	for file in candidate_files:
		try:
			_= is_missing_versetext(file, missing_verses)
			# json_correction(file) 
			# npz_correction(file)
			# npz_test(file)
			# logging.info(f"Edited {file}")
		except Exception as e:
			logging.exception(f"{file} Errored out!!!!")

if __name__ == '__main__':
	main()
