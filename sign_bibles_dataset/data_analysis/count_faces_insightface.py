import argparse
import json
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
    face_count = 0
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(path))
        if img is None:
            return 0, []
        faces = face_analyzer.get(img)
        embeddings.extend([face.embedding for face in faces])
        face_count = len(faces)
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
                face_count += len(faces)
            frame_idx += 1
        cap.release()
    return face_count, embeddings


def cluster_embeddings(embeddings: list[np.ndarray], eps=0.6, min_samples=1):
    if not embeddings:
        return 0
    X = np.vstack(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, labels


def count_unique_people(folder: Path, overwrite=False) -> int:
    log.info(f"Scanning folder: {folder}")
    report_path = folder / "insightface_report.json"
    embed_out_path = folder / "insightface_embeddings.npz"
    if not overwrite and report_path.is_file() and embed_out_path.is_file():
        log.debug("File already exists, skipping!")
        return

    app = FaceAnalysis(
        name="buffalo_l",
        # providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0)

    all_embeddings = []
    face_frame_info = {}

    files = collect_files(folder)
    embedding_sources = []  # one per embedding: {path, frame_idx (optional)}

    for f in tqdm(files, desc="Processing media"):
        face_count, embeddings = extract_faces(app, f)
        if face_count > 0:
            face_frame_info[str(f.resolve())] = face_count
            for emb in embeddings:
                embedding_sources.append(str(f.resolve()))
        all_embeddings.extend(embeddings)

    log.info(f"Total faces found: {len(all_embeddings)}")
    unique_count, labels = cluster_embeddings(all_embeddings)

    log.info(f"Estimated unique individuals: {unique_count}")

    report = {
        "total_faces_detected": len(all_embeddings),
        "unique_individuals": unique_count,
        "files_with_faces": face_frame_info,
        "embedding_sources": embedding_sources,
        "cluster_labels": labels.tolist() if labels is not None else [],
    }

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    np.savez(embed_out_path, embeddings=np.vstack(all_embeddings))

    log.info(f"Saved report to: {report_path.resolve()}")
    log.info(f"Saved embeds to: {embed_out_path.resolve()}")

    return unique_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count unique individuals in media folder using InsightFace.")
    parser.add_argument("folder", type=Path, help="Root folder containing images/videos.")
    args = parser.parse_args()
    count_unique_people(args.folder)
