Using OCR to try and find segments! Works best for the ASL videos. 

1. video_ocr.py will by default check every 10th frame. Outputs json files with names like _frameskip9_*.json
2. rename those to .ocr.json with rename_ocr_files.py 
3. Convert to CSV for easier manual editing with ocr_json_to_ocr_csv.py, saving the results as .ocr.csv
4. Deduplicate lines by keeping only the text _changes_ with deduplicate_ocr_csv_lines.py, outputting .ocr.textchanges.csv
5. Manually go through 



Credit:
* Credit to Nordin Abouzahara for the "look for changes in a ROI and only note when it changes"
* Credit to Amit Moryossef for also coming up with this and providing another implementation 