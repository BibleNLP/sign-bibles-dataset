import json
import difflib
import re
import csv
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

from bible_text_access import get_verses

bsb_file = "BSB_bible.json"
erv_file = "ERV_bible.json"
web_file = "WEB_bible.json"

isl_file = "mat-mark-luke-jhn-list.txt"


bsb_dict = {}
erv_dict = {}
web_dict = {}
isl_list = []

with open(bsb_file, 'r', encoding='utf-8') as f1:
	bsb_dict = json.load(f1)

with open(erv_file, 'r', encoding='utf-8') as f1:
	erv_dict = json.load(f1)

with open(web_file, 'r', encoding='utf-8') as f1:
	web_dict = json.load(f1)


verse_john_pattern = re.compile(r"\d+(\-\d+)?\.MP4")

with open(isl_file, 'r', encoding='utf-8') as fi:
	lines = fi.readlines()
	inp_paths = [itm.split("\t")[1] for itm in lines]
	print(f"{len(inp_paths)=}")
	for remote_path in inp_paths:
		try:
			parts = remote_path.split("/")

			if remote_path.startswith("/John"):
				ref = f"{parts[1]} {parts[2].replace("Ch-", "")}"
				ver_match = re.search(verse_john_pattern, parts[-1])
				if not ver_match:
					raise Exception(f"Cant compose reference from:{parts}")
				verse_parts = ver_match.group().split('.')[0]
			else:
				ref = f"{parts[1]} {parts[2].replace("Ch ", "")}"
				verse = parts[-1].split(".")[0].split(" ")[1]
				verse_parts = "-".join(verse.split("-")[:-1])
			if "-" in verse_parts:
				s_e = verse_parts.split("-")
				verses = range(int(s_e[0]), int(s_e[-1])+1)
			else:
				verses = [int(verse_parts)]
			for v in verses:
				ref = ref.replace("Matthew", "MAT")
				ref = ref.replace("Mark", "MRK")
				ref = ref.replace("Luke", "LUK")
				ref = ref.replace("John", "JHN")
				isl_list.append(f"{ref}:{v}")
		except Exception as exce:
			print(f"Issue in {remote_path=}")
			print(exce)


def align_two_lists(a, b):
	sm = difflib.SequenceMatcher(None, a, b)
	aligned_a, aligned_b = [], []

	for opcode, a0, a1, b0, b1 in sm.get_opcodes():
		if opcode == 'equal':
			aligned_a.extend(a[a0:a1])
			aligned_b.extend(b[b0:b1])
		elif opcode == 'insert':
			aligned_a.extend([''] * (b1 - b0))
			aligned_b.extend(b[b0:b1])
		elif opcode == 'delete':
			aligned_a.extend(a[a0:a1])
			aligned_b.extend([''] * (a1 - a0))
		elif opcode == 'replace':
			len1 = a1 - a0
			len2 = b1 - b0
			maxlen = max(len1, len2)
			aligned_a.extend(a[a0:a1] + [''] * (maxlen - len1))
			aligned_b.extend(b[b0:b1] + [''] * (maxlen - len2))

	return aligned_a, aligned_b

def align_multiple_lists(lists):
    aligned_lists = [lists[0]]

    for i in range(1, len(lists)):
        # Align current aligned result with the next list
        base = aligned_lists[0]
        next_list = lists[i]

        # Align base and next
        new_base, new_next = align_two_lists(base, next_list)

        # Update all previously aligned lists to match the new base length
        for j in range(len(aligned_lists)):
            _, realigned = align_two_lists(new_base, aligned_lists[j])
            aligned_lists[j] = realigned

        aligned_lists.append(new_next)

    return aligned_lists

def versification_analysis():
	print(f"{len(bsb_dict.keys())=}   {len(erv_dict.keys())}")
	print(f"{len(web_dict.keys())=}   {len(erv_dict.keys())=}")

	print(f"{len(isl_list)=}")
	# print(isl_list)


	erv_aligned, web_aligned, bsb_aligned, isl_aligned = align_multiple_lists([list(erv_dict.keys()),
																list(web_dict.keys()),
																list(bsb_dict.keys()),
																isl_list])
	versification_list = [["ERV","WEB","BSB","ISL"]]
	for a, b, c, d in zip(erv_aligned, web_aligned, bsb_aligned, isl_aligned):
		 versification_list.append([a,b,c,d])

	# with open("versification.csv", "w", encoding="utf-8") as fv:
	# 	for row in versification_list:
	# 		fv.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")
	return versification_list

model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast

def semantic_analsis(versification_list):
	erv_web_sim = []
	erv_bsb_sim =[]
	for row in versification_list:
		try:
			if row[0] != "":
				erv_text = get_verses(row[0], bible_name="ERV")
			else:
				erv_text = ""
				# erv_web_sim.append(0)
				# erv_bsb_sim.append(0)
				# continue
			if row[1] != "":
				web_text = get_verses(row[1], bible_name="WEB")
			else:
				web_text = ""
				# erv_web_sim.append(0)
			if row[2] != "":
				bsb_text = get_verses(row[2], bible_name="BSB")
			else:
				bsb_text = ""
				# erv_bsb_sim.append(0)
		except Exception as exce:
			raise Exception(row) from exce
		sentences = [ erv_text, web_text, bsb_text ]
		embeddings = model.encode(sentences, convert_to_tensor=True)
		# if erv_text != "":
		# 	if web_text != "":
		erv_web_sim.append(util.pytorch_cos_sim(embeddings[0], embeddings[1]).cpu().numpy()[0][0])
			# if bsb_text != "":
		erv_bsb_sim.append(util.pytorch_cos_sim(embeddings[0], embeddings[2]).cpu().numpy()[0][0])
		# break
	# print(f"{erv_web_sim=}")
	# print(f"{erv_bsb_sim=}")
	plt.figure(figsize=(15, 5))
	plt.plot(erv_web_sim, label='ERV-WEB', alpha=0.7, linewidth=0.5)
	plt.plot(erv_bsb_sim, label='ERV-BSB', alpha=0.7, linewidth=0.5)

	# Add horizontal lines for expected values (e.g., around 1)
	plt.axhline(1, color='green', linestyle='dotted', linewidth=1)
	plt.axhline(0.9, color='red', linestyle='dashed', linewidth=0.5)
	# plt.axhline(1.1, color='red', linestyle='dashed', linewidth=0.5)

	# Customize the plot
	plt.title('Comparison of Sentence Similarities')
	plt.xlabel('Index')
	plt.ylabel('Similarity Score')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

	print(f"{len(versification_list)=}")
	print(f"{len(erv_web_sim)=}")
	print(f"{len(erv_bsb_sim)=}")

	count = 0
	data = [["Sl. No.", "Ref", "Version", "Similarity", "Sim_range", "ERV-text", "Other-text", "Remarks"]]
	for i in range(len(versification_list)):
		ver_row = versification_list[i]
		w_sim = erv_web_sim[i]
		b_sim = erv_bsb_sim[i]
		count += 1
		erv_text = ""
		if ver_row[0] != "":
			erv_text = get_verses(ver_row[0], bible_name="ERV").replace("\t","").replace("\n","")
		web_text = ""
		if ver_row[1] != "":
			web_text = get_verses(ver_row[1], bible_name="WEB").replace("\t","").replace("\n","")
		row = [str(count), ver_row[0], "WEB", str(w_sim), simalarity_range(w_sim), erv_text, web_text, ""]
		data.append(row)
		count += 1
		erv_text = ""
		if ver_row[0] != "":
			erv_text = get_verses(ver_row[0], bible_name="ERV").replace("\t","").replace("\n","")
		bsb_text = ""
		if ver_row[2] != "":
			bsb_text = get_verses(ver_row[2], bible_name="BSB").replace("\t","").replace("\n","")
		row = [str(count), ver_row[0], "BSB", str(b_sim), simalarity_range(b_sim), erv_text, bsb_text, ""]
		data.append(row)
	with open('semantic_analysis.csv', 'w', newline='', encoding='utf-8') as file:
	    writer = csv.writer(file,delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
	    writer.writerows(data)

def simalarity_range(sim):
	if sim > 0.95:
		return "1-0.95"
	if sim > 0.90:
		return "0.95-0.90"
	if sim > 0.80:
		return "0.9-0.8"
	if sim > 0.7:
		return "0.8-0.7"
	if sim > 0.5:
		return "0.7-0.5"
	if sim > 0.3:
		return "0.5-0.3"
	if sim > 0.1:
		return "0.3-0.1"
	return "0.1-0"


if __name__ == '__main__':
	versification_list = versification_analysis()

	semantic_analsis(versification_list[23146:26925])