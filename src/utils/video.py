"""
Video processing module for the vgg-face-reg project.

This module provides functionality to read raw video files and extract frames
at a specified dynamic sampling rate to prevent temporal correlation while
accommodating varying video lengths. It includes CLI support for batch processing.
"""

import argparse
import cv2
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Configure logging for the CLI output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_frames_dynamically(
    video_path: Path, output_dir: Path, frames_per_second_to_extract: float = 3.0
) -> Optional[int]:
    """
    Extracts frames from a video file based on a dynamic sampling rate.

    Calculates a mathematical stride based on the source video's native FPS
    to extract a specific number of frames per second of footage.

    Args:
        video_path (Path): The absolute or relative path to the raw .mp4 file.
        output_dir (Path): The directory where extracted .jpg frames will be saved.
        frames_per_second_to_extract (float): The desired number of frames to
            extract per second of video. Defaults to 3.0.

    Returns:
        Optional[int]: The total number of frames successfully extracted,
            or None if the video could not be opened.

    Raises:
        ValueError: If `frames_per_second_to_extract` is less than or equal to 0.
    """
    if frames_per_second_to_extract <= 0:
        raise ValueError("frames_per_second_to_extract must be greater than 0.")

    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        return None

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OpenCV VideoCapture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video stream: {video_path}")
        return None

    # Retrieve native video properties
    native_fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        logging.warning(
            f"Could not determine native FPS for {video_path}. Defaulting to 30.0."
        )
        native_fps = 30.0

    # Calculate the temporal stride
    stride: int = max(1, int(round(native_fps / frames_per_second_to_extract)))

    logging.info(
        f"Processing {video_path.name}: Native FPS={native_fps:.2f}, Stride={stride}"
    )

    frame_index: int = 0
    extracted_count: int = 0

    # Initialize tqdm progress bar
    with tqdm(
        total=total_frames, desc=f"Extracting {video_path.stem}", unit="frame"
    ) as pbar:
        while True:
            success, frame = cap.read()

            if not success:
                break  # End of video stream

            # Apply modulo arithmetic for dynamic sampling
            if frame_index % stride == 0:
                # Format: memberName_frameIndex.jpg (e.g., Hung_0045.jpg)
                member_name: str = video_path.stem
                out_filename: str = f"{member_name}_{frame_index:04d}.jpg"
                out_path: Path = output_dir / out_filename

                cv2.imwrite(str(out_path), frame)
                extracted_count += 1

            frame_index += 1
            pbar.update(1)  # Advance the progress bar by 1 frame

    cap.release()
    logging.info(
        f"Successfully extracted {extracted_count} frames for {video_path.stem}."
    )

    return extracted_count


def main() -> None:
    """
    CLI entry point to batch process a directory of video files.
    """
    parser = argparse.ArgumentParser(
        description="Dynamically extract frames from video files."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw .mp4 files (default: data/raw).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save the extracted .jpg frames (default: data/processed).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Target frames to extract per second of video.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        logging.error(
            f"Input directory does not exist or is not a directory: {args.input_dir}"
        )
        return

    video_files = list(args.input_dir.glob("*.mp4"))
    if not video_files:
        logging.warning(f"No .mp4 files found in {args.input_dir}")
        return

    logging.info(f"Found {len(video_files)} video(s) to process.")

    total_extracted = 0
    for video_path in video_files:
        count = extract_frames_dynamically(video_path, args.output_dir, args.fps)
        if count:
            total_extracted += count

    logging.info(
        f"Batch processing complete. Total frames extracted across all videos: {total_extracted}"
    )


if __name__ == "__main__":
    main()
