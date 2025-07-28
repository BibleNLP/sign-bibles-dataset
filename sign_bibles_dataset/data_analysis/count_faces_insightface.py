import logging
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("face_counter")

# Supported image and video formats
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_files(root: Path) -> list[Path]:
    return [
        f
        for f in root.rglob("*")
        if f.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS) and not any(p.name == "skintone" for p in f.parents)
    ]


def extract_faces(face_analyzer, path: Path, every_n_frames=10):
    embeddings = []
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(path))
        if img is None:
            return []
        faces = face_analyzer.get(img)
        embeddings.extend([face.embedding for face in faces])
    elif path.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n_frames == 0:
                faces = face_analyzer.get(frame)
                embeddings.extend([face.embedding for face in faces])
            frame_idx += 1
        cap.release()
    return embeddings


def cluster_embeddings(embeddings: list[np.ndarray], eps=0.6, min_samples=2):
    if not embeddings:
        return 0
    X = np.vstack(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters


def count_unique_people(folder: Path):
    log.info(f"Scanning folder: {folder}")
    app = FaceAnalysis(
        name="buffalo_l",
        # providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0)

    all_embeddings = []
    files = collect_files(folder)
    for f in tqdm(files, desc="Processing media"):
        try:
            embeddings = extract_faces(app, f)
            all_embeddings.extend(embeddings)
        except Exception as e:
            log.warning(f"Error processing {f}: {e}")

    log.info(f"Total faces found: {len(all_embeddings)}")
    unique_count = cluster_embeddings(all_embeddings)
    log.info(f"Estimated unique individuals: {unique_count}")
    return unique_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count unique individuals in media folder using InsightFace.")
    parser.add_argument("folder", type=Path, help="Root folder containing images/videos.")
    args = parser.parse_args()
    count_unique_people(args.folder)
