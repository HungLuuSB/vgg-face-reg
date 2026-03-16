"""
Metric evaluation and pair generation module for the vgg-face-reg project.

This script uses MTCNN 2D facial landmarks to estimate head pose (yaw and pitch),
categorizes frames, and systematically generates mathematically challenging
Genuine and Imposter pairs for open-set verification testing.
"""

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# Import the detector to leverage its MTCNN instance
from core.detector import FaceDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def calculate_pose_ratios(landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the 2D Yaw and Pitch ratios from 5-point facial landmarks.

    Args:
        landmarks (np.ndarray): Array of shape (5, 2) containing (x, y) coordinates for:
            [0]: Left Eye, [1]: Right Eye, [2]: Nose, [3]: Left Mouth, [4]: Right Mouth.
            Note: "Left Eye" refers to the eye on the left side of the image.

    Returns:
        Tuple[float, float]: The calculated (yaw_ratio, pitch_ratio).
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # Calculate Yaw Ratio (Rx)
    # R_yaw = (Nose_x - LeftEye_x) / (RightEye_x - Nose_x)
    numerator_yaw = nose[0] - left_eye[0]
    denominator_yaw = right_eye[0] - nose[0]

    # Prevent division by zero if face is perfectly profile
    if denominator_yaw == 0:
        denominator_yaw = 0.001
    yaw_ratio = numerator_yaw / denominator_yaw

    # Calculate Pitch Ratio (Ry)
    # R_pitch = (Nose_y - AvgEye_y) / (AvgMouth_y - Nose_y)
    avg_eye_y = (left_eye[1] + right_eye[1]) / 2.0
    avg_mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0

    numerator_pitch = nose[1] - avg_eye_y
    denominator_pitch = avg_mouth_y - nose[1]

    if denominator_pitch == 0:
        denominator_pitch = 0.001
    pitch_ratio = numerator_pitch / denominator_pitch

    return float(yaw_ratio), float(pitch_ratio)


def categorize_pose(yaw_ratio: float, pitch_ratio: float) -> str:
    """
    Assigns a categorical pose label based on geometric ratios.

    Args:
        yaw_ratio (float): The horizontal alignment ratio.
        pitch_ratio (float): The vertical alignment ratio.

    Returns:
        str: A categorical string label representing the pose.
    """
    if yaw_ratio < 0.75:
        return "Yaw_Left"
    elif yaw_ratio > 1.25:
        return "Yaw_Right"
    elif pitch_ratio < 0.80:
        return "Pitch_Up"
    elif pitch_ratio > 1.20:
        return "Pitch_Down"
    else:
        return "Frontal"


def generate_evaluation_pairs(
    image_dir: Path, num_genuine: int = 500, num_imposter: int = 500
) -> List[Tuple[str, str, int, str]]:
    """
    Processes a directory of images, categorizes their pose, and builds pairs.

    Args:
        image_dir (Path): Directory containing the extracted .jpg frames.
        num_genuine (int): Target number of Genuine pairs to generate.
        num_imposter (int): Target number of Imposter pairs to generate.

    Returns:
        List[Tuple[str, str, int, str]]: A list of tuples containing:
            (Image_A_Path, Image_B_Path, Label[1=Genuine, 0=Imposter], Pose_Variance_Type)
    """
    detector = FaceDetector(image_size=224)

    # Dictionary structure: identity -> pose_category -> list of file paths
    # Example: {"Hung": {"Frontal": [path1, path2], "Yaw_Left": [path3]}}
    identity_pose_map: Dict[str, Dict[str, List[Path]]] = {}

    image_paths = list(image_dir.glob("*.jpg"))
    logging.info(f"Analyzing {len(image_paths)} images for pose estimation...")

    # 1. Analyze and Tag Every Image
    for img_path in tqdm(image_paths, desc="Tagging Poses"):
        identity = img_path.stem.split("_")[0]

        try:
            img_pil = Image.open(str(img_path)).convert("RGB")
            # Bypass the wrapper to get raw landmarks from the MTCNN instance
            boxes, probs, landmarks = detector.mtcnn.detect(img_pil, landmarks=True)

            if landmarks is not None and len(landmarks) > 0:
                # Use the landmarks of the first detected face
                yaw, pitch = calculate_pose_ratios(landmarks[0])
                pose_tag = categorize_pose(yaw, pitch)

                if identity not in identity_pose_map:
                    identity_pose_map[identity] = {}
                if pose_tag not in identity_pose_map[identity]:
                    identity_pose_map[identity][pose_tag] = []

                identity_pose_map[identity][pose_tag].append(img_path)
        except Exception as e:
            continue  # Skip unreadable or faceless images

    pairs_ledger: List[Tuple[str, str, int, str]] = []
    identities = list(identity_pose_map.keys())

    if len(identities) < 2:
        logging.error("Need at least 2 distinct identities to generate Imposter pairs.")
        return pairs_ledger

    # 2. Generate Genuine Pairs (Hard Mode: Frontal vs Non-Frontal)
    logging.info("Generating Genuine Pairs...")
    attempts = 0
    while len(pairs_ledger) < num_genuine and attempts < num_genuine * 10:
        attempts += 1
        identity = random.choice(identities)
        poses = list(identity_pose_map[identity].keys())

        # We need at least 'Frontal' and one other pose for a hard pair
        if "Frontal" in poses and len(poses) > 1:
            non_frontal_pose = random.choice([p for p in poses if p != "Frontal"])

            if (
                identity_pose_map[identity]["Frontal"]
                and identity_pose_map[identity][non_frontal_pose]
            ):
                img_a = random.choice(identity_pose_map[identity]["Frontal"])
                img_b = random.choice(identity_pose_map[identity][non_frontal_pose])

                pairs_ledger.append(
                    (
                        str(img_a.name),
                        str(img_b.name),
                        1,
                        f"Frontal_vs_{non_frontal_pose}",
                    )
                )

    # 3. Generate Imposter Pairs (Different Identities)
    logging.info("Generating Imposter Pairs...")
    attempts = 0
    while (
        len(pairs_ledger) < num_genuine + num_imposter and attempts < num_imposter * 10
    ):
        attempts += 1
        id_a, id_b = random.sample(identities, 2)

        # Get all flat paths for these identities
        paths_a = [p for sublist in identity_pose_map[id_a].values() for p in sublist]
        paths_b = [p for sublist in identity_pose_map[id_b].values() for p in sublist]

        if paths_a and paths_b:
            img_a = random.choice(paths_a)
            img_b = random.choice(paths_b)
            pairs_ledger.append(
                (str(img_a.name), str(img_b.name), 0, "Identity_Mismatch")
            )

    return pairs_ledger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Metric Pair Generation via MTCNN."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the extracted frames.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=Path("data/evaluation/pairs_ledger.csv"),
        help="Path to save the generated pairs ledger.",
    )
    parser.add_argument(
        "--genuine", type=int, default=1000, help="Number of Genuine pairs."
    )
    parser.add_argument(
        "--imposter", type=int, default=1000, help="Number of Imposter pairs."
    )
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = generate_evaluation_pairs(args.input_dir, args.genuine, args.imposter)

    if pairs:
        with open(args.output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image_A", "Image_B", "Is_Genuine", "Variance_Type"])
            writer.writerows(pairs)
        logging.info(
            f"Successfully wrote {len(pairs)} evaluation pairs to {args.output_csv}"
        )
    else:
        logging.warning("Failed to generate pairs. Check dataset size and variance.")


if __name__ == "__main__":
    main()
