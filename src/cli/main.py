"""
Main Command Line Interface for the vgg-face-reg project.

Supports BOTH Approach 1 (Zero-Shot Emebedding + k-NN) and
Approach 2 (Fine-Tuned Softmax Classification) via the --approach flag.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.detector import FaceDetector
from core.embedder import VGGFaceEmbedder
from core.matcher import FaceMatcher

# Import required for Approach 2
from utils.train_finetune import setup_finetuning_model, ExtractedFaceDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time Face Verification CLI")

    parser.add_argument(
        "--approach",
        type=str,
        choices=["zeroshot", "finetuned"],
        default="zeroshot",
        help="Select the architecture: 'zeroshot' (k-NN) or 'finetuned' (Softmax).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the .pth weights file (vgg_face_dag.pth or vgg_face_finetuned.pth).",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("data/gallery/reference.pt"),
        help="[Approach 1 Only] Path to the saved reference gallery .pt file.",
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=Path("data/processed"),
        help="[Approach 2 Only] Path to training images to extract class labels.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Threshold (Cosine Similarity for zeroshot, Confidence % for finetuned).",
    )

    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("Initializing Face Detector (MTCNN)...")
    detector = FaceDetector(image_size=224, device=device)

    # ==========================================
    # BRANCHING LOGIC: LOAD ARCHITECTURE
    # ==========================================
    if args.approach == "zeroshot":
        logging.info("--> MODE: Approach 1 (Zero-Shot Feature Extraction + k-NN)")
        embedder = VGGFaceEmbedder(weights_path=args.weights, device=device)
        matcher = FaceMatcher(gallery_path=args.gallery, threshold=args.threshold, k=5)

    elif args.approach == "finetuned":
        logging.info("--> MODE: Approach 2 (Fine-Tuned Softmax Classification)")
        # We need the dataset just to get the list of 5 member names (classes)
        train_dataset = ExtractedFaceDataset(
            image_dir=args.train_dir, detector=detector
        )
        classes = train_dataset.classes

        # Load Original DAG weights first to build the architecture
        model = setup_finetuning_model(
            Path("models/vgg_face_dag.pth"), len(classes), device
        )
        # Overwrite with the Finetuned weights
        model.load_state_dict(
            torch.load(args.weights, map_location=device, weights_only=True)
        )
        model.eval()

    # ==========================================
    # START CAMERA
    # ==========================================
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logging.info("Starting live inference loop. Press 'q' to exit.")
    prev_time: float = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time: float = time.time()
        fps: float = (
            1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        )
        prev_time = current_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Detect Face
        boxes, _ = detector.mtcnn.detect(pil_frame)
        face_tensor: Optional[torch.Tensor] = detector.process_frame(pil_frame)

        display_name: str = "Searching..."
        color: Tuple[int, int, int] = (255, 255, 255)

        if face_tensor is not None and boxes is not None:
            # ==========================================
            # INFERENCE LOGIC BASED ON APPROACH
            # ==========================================
            if args.approach == "zeroshot":
                # Approach 1: Metric Learning EER
                live_embedding = embedder.get_embedding(face_tensor)
                identity, score = matcher.identify(live_embedding)

            elif args.approach == "finetuned":
                # Approach 2: Softmax Classification
                face_tensor = face_tensor.to(device)
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    outputs = model(face_tensor)
                    # Convert raw logits to percentages (0.0 to 1.0)
                    probabilities = F.softmax(outputs, dim=1)
                    max_prob, predicted_idx = torch.max(probabilities.data, 1)

                    score = max_prob.item()

                    # Apply Softmax confidence threshold
                    if score >= args.threshold:
                        identity = classes[predicted_idx.item()]
                    else:
                        identity = "Unknown"

            # ==========================================
            # DRAW GUI
            # ==========================================
            if identity == "Unknown":
                display_name = f"Unknown ({score:.2f})"
                color = (0, 0, 255)  # Red
            else:
                display_name = f"{identity} ({score:.2f})"
                color = (0, 255, 0)  # Green

            box = boxes[0]
            cv2.rectangle(
                frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
            )
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1]) - 35),
                (int(box[2]), int(box[1])),
                color,
                cv2.FILLED,
            )
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

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Mode: {args.approach}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("VGG-Face Real-Time Verification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
