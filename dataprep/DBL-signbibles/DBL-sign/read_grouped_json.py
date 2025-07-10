import json
from pathlib import Path

import pandas as pd

JSON_PATH = "/opt/home/cleong/projects/pose-evaluation/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/video_file_passages_grouped_with_ebible_vrefs_and_text.json"

with open(JSON_PATH, encoding="utf-8") as f:
    grouped_data = json.load(f)

# Flatten with pandas.json_normalize
df = pd.json_normalize(
    grouped_data, record_path="videos", meta=["language_code", "version_name", "dbl_id"]
)

# Optional: reorder columns
# df = df[["language_code", "version_name", "dbl_id", "mp4_path", "bible_passage", "ebible_vref_indices"]]

print(df.head())
out = f"{Path(JSON_PATH).stem}_ungrouped.csv"
df.to_csv(out, index=False)
