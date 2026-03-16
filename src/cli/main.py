"""
Main Command Line Interface for the vgg-face-reg project.

This script initializes the FaceDetector, VGGFaceEmbedder, and FaceMatcher.
It captures the live webcam feed via OpenCV, extracts faces, computes their
embeddings, and overlays the matched identity in real-time.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Import the core modules developed previously
from core.detector import FaceDetector
from core.embedder import VGGFaceEmbedder
from core.matcher import FaceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the application.

    Returns:
        argparse.Namespace: The populated namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Real-Time Open-Set Face Verification CLI"
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the pre-trained VGGFace .pth weights file.",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        required=True,
        help="Path to the saved reference gallery .pt file.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index (default: 0 for laptop webcam).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine Similarity threshold for Unknown rejection (tau).",
    )

    return parser.parse_args()


def main() -> None:
    """
    The main execution loop for the real-time face recognition pipeline.
    """
    args: argparse.Namespace = parse_arguments()

    # 1. Initialize Core Subsystems
    logging.info("Initializing neural network subsystems...")

    detector = FaceDetector(image_size=224)
    embedder = VGGFaceEmbedder(weights_path=args.weights)
    matcher = FaceMatcher(gallery_path=args.gallery, threshold=args.threshold, k=5)

    # 2. Initialize Video Capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logging.error(f"Failed to open camera index {args.camera}.")
        return

    # Set camera resolution to 720p for a balance of clarity and processing speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logging.info("Starting live inference loop. Press 'q' to exit.")

    # Variables for FPS calculation
    prev_time: float = 0.0

    while True:
        success: bool
        frame: cv2.typing.MatLike
        success, frame = cap.read()

        if not success:
            logging.error("Failed to read frame from camera. Exiting loop.")
            break

        # Calculate FPS
        current_time: float = time.time()
        fps: float = (
            1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        )
        prev_time = current_time

        # Convert OpenCV BGR frame to RGB PIL Image for MTCNN processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # 3. Detect and Align Face
        # We also extract the raw bounding box to draw the visual rectangle
        boxes, _ = detector.mtcnn.detect(pil_frame)
        face_tensor: Optional[torch.Tensor] = detector.process_frame(pil_frame)

        display_name: str = "Searching..."
        color: Tuple[int, int, int] = (255, 255, 255)  # White default

        if face_tensor is not None and boxes is not None:
            # 4. Extract Embedding
            live_embedding: torch.Tensor = embedder.get_embedding(face_tensor)

            # 5. Perform Metric Matching
            identity, score = matcher.identify(live_embedding)

            # 6. Format Output
            if identity == "Unknown":
                display_name = f"Unknown ({score:.2f})"
                color = (0, 0, 255)  # Red for Unknown (BGR format)
            else:
                display_name = f"{identity} ({score:.2f})"
                color = (0, 255, 0)  # Green for Known Member

            # Draw the bounding box for the first detected face
            box = boxes[0]
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, start_point, end_point, color, 2)

            # Draw the label background
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1]) - 35),
                (int(box[2]), int(box[1])),
                color,
                cv2.FILLED,
            )

            # Draw the text
            cv2.putText(
                frame,
                display_name,
                (int(box[0]) + 5, int(box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Draw FPS counter
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 7. Render GUI
        cv2.imshow("VGG-Face Real-Time Verification", frame)

        # Exit condition: Wait 1ms for the 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Application terminated cleanly.")


if __name__ == "__main__":
    main()
