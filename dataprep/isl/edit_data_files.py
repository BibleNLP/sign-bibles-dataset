import json
import logging
import argparse
from pathlib import Path
import numpy as np

logging.basicConfig(filename='logs/app.log', level=logging.INFO,
					format='%(asctime)s - %(levelname)s - %(message)s')

def json_correction(filename):
	orig_data = {}
	with open(filename, 'r', encoding='utf-8') as fp:
		orig_data = json.load(fp)
	orig_data['transcripts'][0]['language']['BCP-47'] = "en-US"
	orig_data['glosses'][0]['language']['BCP-47'] = "en-US"
	with open(filename, 'w', encoding='utf-8') as fp:
		json.dump(orig_data, fp, indent=4)

def npz_correction(filename):
	old_obj = np.load(filename, allow_pickle=True)
	new_array = []
	for item in old_obj['frames']:
		if len(item) < 1:
			item = [np.full(134, np.nan, dtype=np.float64)]
		elif len(item > 1):
			item = [item[0]]
		assert len(item) == 1, f"{len(item)=}"
		assert len(item[0]) == 134, f"{len(item[0])=}"
		new_array.append(item)
	np.savez_compressed(filename, frames=np.array(new_array, dtype=np.float64))

def npz_test(filename):
	new_obj = np.load(filename) # load w/o pickle
	assert 'frames' in new_obj
	assert len(new_obj['frames'][0][0]) == 134
	assert len(new_obj['frames'][0][0][0]) == 2


def main():
	parser = argparse.ArgumentParser(description='Calculate total duration from JSON files.')
	parser.add_argument('directories', nargs='+', help='List of directories to search')
	args = parser.parse_args()

	candidate_files = [
		file 
		for directory in args.directories 
		# for file in Path(directory).rglob("*.json") 
		for file in Path(directory).rglob("*.npz")
	]

	for file in candidate_files:
		try:
			# json_correction(file) 
			# npz_correction(file)
			npz_test(file)
			logging.info(f"Edited {file}")
		except Exception as e:
			logging.exception(f"{file} Errored out!!!!")

if __name__ == '__main__':
	main()
