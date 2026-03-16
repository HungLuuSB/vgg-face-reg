"""
Gallery building utility for the vgg-face-reg project.

This script iterates through a directory of extracted face images, detects the
faces using MTCNN, extracts the 4096-dimensional embeddings using VGGFace,
and compiles them into a single serialized PyTorch dictionary for real-time inference.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from core.detector import FaceDetector
from core.embedder import VGGFaceEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the reference gallery for Face Matching."
    )
    parser.add_argument(
        "-i",
        "--images_dir",
        type=Path,
        default=Path("data/processed"),
        required=True,
        help="Directory containing the processed .jpg images.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        default=Path("models/vgg_face_dag.pth"),
        required=True,
        help="Path to the VGGFace .pth weights.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/gallery/reference.pt"),
        required=True,
        help="Path to save the output reference.pt file.",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_arguments()

    if not args.images_dir.exists():
        logging.error(f"Image directory not found: {args.images_dir}")
        return

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Subsystems
    logging.info("Loading models into memory...")
    detector = FaceDetector(image_size=224)
    embedder = VGGFaceEmbedder(weights_path=args.weights)

    embeddings_list: List[torch.Tensor] = []
    labels_list: List[str] = []

    # 2. Iterate over all JPG images in the directory
    # We assume images are named in the format "MemberName_001.jpg"
    image_paths: List[Path] = list(args.images_dir.glob("*.jpg"))

    if not image_paths:
        logging.error(f"No .jpg files found in {args.images_dir}")
        return

    logging.info(f"Found {len(image_paths)} images. Processing...")

    for img_path in image_paths:
        try:
            # Extract the identity label from the filename (everything before the first underscore)
            label: str = img_path.stem.split("_")[0]

            # Detect and align the face
            face_tensor: Optional[torch.Tensor] = detector.process_frame(img_path)

            if face_tensor is None:
                logging.warning(f"No face detected in {img_path.name}. Skipping.")
                continue

            # Extract the 4096-D embedding
            embedding: torch.Tensor = embedder.get_embedding(face_tensor)

            embeddings_list.append(embedding)
            labels_list.append(label)

        except Exception as e:
            logging.error(f"Failed to process {img_path.name}: {e}")

    # 3. Compile and Save the Gallery
    if not embeddings_list:
        logging.error("No valid faces were processed. Gallery not created.")
        return

    # Stack the list of (1, 4096) tensors into a single (N, 4096) matrix
    final_embeddings: torch.Tensor = torch.cat(embeddings_list, dim=0)

    gallery_dict: Dict[str, any] = {
        "embeddings": final_embeddings,
        "labels": labels_list,
    }

    torch.save(gallery_dict, str(args.output))
    logging.info(f"Successfully saved {len(labels_list)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
