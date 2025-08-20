import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import math

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


def extract_faces(face_analyzer, path: Path, every_n_frames=1):
    embeddings = []
    face_images = []  # store cropped faces
    face_count = 0
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(path))
        if img is None:
            return 0, [], []
        faces = face_analyzer.get(img)
        for face in faces:
            embeddings.append(face.embedding)
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                face_images.append(crop)
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
                for face in faces:
                    embeddings.append(face.embedding)
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        face_images.append(crop)
                face_count += len(faces)
            frame_idx += 1
        cap.release()
    return face_count, embeddings, face_images


def cluster_embeddings(embeddings: list[np.ndarray], eps=0.6, min_samples=1):
    if not embeddings:
        return 0, None  
    X = np.vstack(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, labels


def make_cluster_grid(face_images, labels, output_path="insightface_grid.png", size=128):
    """Stitch one representative face per cluster into a grid."""
    if labels is None or len(face_images) == 0:
        return

    clusters = {}
    for img, label in zip(face_images, labels):
        if label == -1:
            continue  # skip noise
        if label not in clusters:  # keep first representative per cluster
            clusters[label] = img

    reps = list(clusters.values())
    if not reps:
        return

    # resize to same size
    reps = [cv2.resize(img, (size, size)) for img in reps]

    n = len(reps)
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)

    # pad with black if not square
    while len(reps) < grid_cols * grid_rows:
        reps.append(np.zeros((size, size, 3), dtype=np.uint8))

    # build grid
    rows = []
    for r in range(grid_rows):
        row_imgs = reps[r * grid_cols:(r + 1) * grid_cols]
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)

    cv2.imwrite(output_path, grid)
    log.info(f"Saved cluster grid to: {output_path}")


def count_unique_people(folder: Path, overwrite=False) -> int:
    log.info(f"Scanning folder: {folder}")
    report_path = folder / "insightface_report.json"
    embed_out_path = folder / "insightface_embeddings.npz"
    grid_out_path = folder / "insightface_grid.png"

    if not overwrite and report_path.is_file() and embed_out_path.is_file() and grid_out_path.is_file():
        log.debug("Outputs already exist, skipping!")
        return

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)

    all_embeddings = []
    all_face_images = []
    face_frame_info = {}
    embedding_sources = []

    files = collect_files(folder)
    for f in tqdm(files, desc="Processing media"):
        face_count, embeddings, face_images = extract_faces(app, f)
        if face_count > 0:
            face_frame_info[str(f.resolve())] = face_count
            for emb in embeddings:
                embedding_sources.append(str(f.resolve()))
        all_embeddings.extend(embeddings)
        all_face_images.extend(face_images)

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

    if all_embeddings:
        np.savez(embed_out_path, embeddings=np.vstack(all_embeddings))

    make_cluster_grid(all_face_images, labels, output_path=str(grid_out_path))

    return unique_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count unique individuals in media folder using InsightFace.")
    parser.add_argument("folder", type=Path, help="Root folder containing images/videos.")
    args = parser.parse_args()
    count_unique_people(args.folder)
