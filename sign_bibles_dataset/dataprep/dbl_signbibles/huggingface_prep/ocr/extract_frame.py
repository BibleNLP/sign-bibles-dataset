import argparse
import logging
import tempfile
from pathlib import Path

import cv2
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import logging

log = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_skip: int,
    min_frame: int,
    max_frame: int | None,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(min_frame, 0)
    end = min(max_frame if max_frame is not None else total_frames, total_frames)

    if start >= end:
        raise ValueError(f"Invalid frame range: min_frame={min_frame}, max_frame={max_frame}, total={total_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    current_frame = start
    frames_to_process = end - start

    log.info(f"Extracting frames from {video_path.name} to {output_dir}")
    log.info(f"Frame range: [{start}, {end}), skipping every {frame_skip} frames")

    # Disable tqdm if logger is too quiet
    disable_tqdm = not log.isEnabledFor(logging.INFO)
    pbar = tqdm(total=frames_to_process, desc="Extracting", disable=disable_tqdm)

    while current_frame < end:
        ret, frame = cap.read()
        if not ret:
            log.warning(f"Failed to read frame at index {current_frame}")
            break

        if frame_skip == 0 or (current_frame - start) % (frame_skip + 1) == 0:
            frame_path = output_dir / f"frame_{current_frame:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)

        current_frame += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    log.info(f"Saved {len(frame_paths)} frames to {output_dir}")
    return frame_paths


def handle_same_folder(video_path: Path, **kwargs):
    return extract_frames(video_path, video_path.parent, **kwargs)


def handle_subfolder(video_path: Path, **kwargs):
    subfolder = video_path.parent / (video_path.stem + "_frames")
    return extract_frames(video_path, subfolder, **kwargs)


def handle_temp(video_path: Path, **kwargs):
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_frames_"))
    paths = extract_frames(video_path, temp_dir, **kwargs)
    print(f"\n[Temp Mode] Saved to: {temp_dir}")
    print("Sample frames:")
    for path in paths[:5]:
        print(f"  {path}")
    input("Press Enter to continue...")
    return paths


def handle_output_path(video_path: Path, output_path: Path, **kwargs):
    output_path.mkdir(parents=True, exist_ok=True)
    return extract_frames(video_path, output_path, **kwargs)


def parse_step(step_arg: str, fps: float) -> int:
    try:
        return int(step_arg)
    except ValueError:
        if step_arg.endswith("s"):
            seconds = float(step_arg[:-1])
            frame_skip = round(fps * seconds) - 1
            return max(0, frame_skip)
    raise ValueError(f"Invalid --step: {step_arg!r}. Use int or seconds (e.g. '1s')")


def parse_frame_index(arg: str | None, fps: float, total_frames: int) -> int | None:
    if arg is None:
        return None
    try:
        return int(arg)
    except ValueError:
        if isinstance(arg, str) and arg.endswith("s"):
            seconds = float(arg[:-1])
            frame = round(fps * seconds)
            return min(frame, total_frames)
    raise ValueError(f"Invalid frame index: {arg!r}. Use int or seconds (e.g. '3.5s')")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from video files.")
    parser.add_argument("video_path", type=Path, help="Path to video file (e.g., foo.mp4)")
    parser.add_argument(
        "--mode",
        choices=["same_folder", "subfolder", "temp", "output_path"],
        # required=True,
        default="subfolder",
        help="Where to save extracted frames",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Target directory (required for mode 'output_path')",
    )
    parser.add_argument(
        "--step",
        default="0",
        help="Frame step: integer (e.g., '4') or time (e.g., '1s', '0.5s') [default: 0 = every frame]",
    )
    parser.add_argument(
        "--min",
        default="0",
        help="Start from this frame or second (e.g., '100' or '3.5s')",
    )
    parser.add_argument(
        "--max",
        default=None,
        help="Stop before this frame or second (e.g., '300' or '10s')",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't output things!",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    video_path = args.video_path.resolve()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    kwargs = {
        "frame_skip": parse_step(args.step, fps),
        "min_frame": parse_frame_index(args.min, fps, total_frames) or 0,
        "max_frame": parse_frame_index(args.max, fps, total_frames),
    }

    if args.mode == "same_folder":
        handle_same_folder(video_path, **kwargs)
    elif args.mode == "subfolder":
        handle_subfolder(video_path, **kwargs)
    elif args.mode == "temp":
        handle_temp(video_path, **kwargs)
    elif args.mode == "output_path":
        if not args.output_path:
            raise ValueError("--output_path is required when mode is 'output_path'")
        handle_output_path(video_path, args.output_path.resolve(), **kwargs)


if __name__ == "__main__":
    main()
