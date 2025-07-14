from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("OCR Manual Edit Frame Viewer")

# --- Step 1: Input directory ---
directory = st.text_input("Enter the path to your directory:", value=".")

if not directory:
    st.warning("Please enter a directory.")
    st.stop()

root = Path(directory).expanduser().resolve()
if not root.is_dir():
    st.error(f"{root} is not a valid directory.")
    st.stop()

# --- Step 2: Find all .mp4 videos with .ocr.manualedit.csv counterparts ---
videos = sorted(root.glob("*.mp4"))
video_stems = [v.stem for v in videos if (root / f"{v.stem}.ocr.manualedit.csv").exists()]

if not video_stems:
    st.error("No .mp4 + .ocr.manualedit.csv pairs found.")
    st.stop()

# --- Step 3: Track selected video index ---
if "video_index" not in st.session_state:
    st.session_state.video_index = 0

# Keep index in bounds in case directory changed
if st.session_state.video_index >= len(video_stems):
    st.session_state.video_index = 0

# --- Step 4: Show dropdown (binds to session state index) ---
selected_stem = st.selectbox(
    "Select a video:",
    video_stems,
    index=st.session_state.video_index,
    key="video_dropdown",
)

# Update session state index when dropdown changes
st.session_state.video_index = video_stems.index(selected_stem)

# --- Step 5: Navigation buttons ---
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("⬅️ Previous"):
        st.session_state.video_index = (st.session_state.video_index - 1) % len(video_stems)

with col3:
    if st.button("Next ➡️"):
        st.session_state.video_index = (st.session_state.video_index + 1) % len(video_stems)

# --- Step 6: Load CSV ---
csv_path = root / f"{selected_stem}.ocr.manualedit.csv"

try:
    df = pd.read_csv(csv_path, dtype={"frame_index": int, "text": str})
    df = df.sort_values("frame_index")
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

st.success(f"Loaded `{csv_path.name}` with {len(df)} rows.")

# --- Step 7: Display text + thumbnail per row ---
width = st.number_input("Custom image width (0 for full column)", min_value=0, value=200)
for _, row in df.iterrows():
    frame_index = row["frame_index"]
    text = row["text"]
    image_path = root / f"{selected_stem}.frame{frame_index}.png"

    cols = st.columns([1, 3])
    with cols[0]:
        st.markdown(f"**Frame {frame_index}**")
        st.markdown(f"`{text}`")

    with cols[1]:
        if image_path.exists():
            if not width:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)
            else:
                st.image(str(image_path), caption=image_path.name, width=width)
        else:
            st.warning(f"Missing: {image_path.name}")
