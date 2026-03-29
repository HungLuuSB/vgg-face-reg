"""
Testing module for the Fine-Tuned VGG-Face (Approach 2).

This script loads the custom fine-tuned weights and evaluates the closed-set
accuracy on the independent evaluation dataset.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_finetune import ExtractedFaceDataset, setup_finetuning_model
from core.detector import FaceDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Test Fine-Tuned VGG-Face")
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=Path("data/processed"),
        help="Used to map class indices reliably",
    )
    parser.add_argument("--test_dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument(
        "--weights", type=Path, default=Path("models/vgg_face_finetuned.pth")
    )
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not args.weights.exists():
        logging.error(
            f"Fine-tuned weights not found at {args.weights}. Please train and save first."
        )
        return

    # 1. Initialize Detector
    detector = FaceDetector(image_size=224, device=device)

    # 2. Get original class mapping from Training data to ensure index matches
    logging.info("Loading class mapping from training directory...")
    train_dataset = ExtractedFaceDataset(image_dir=args.train_dir, detector=detector)
    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    # 3. Load Test Data
    logging.info(f"Loading test data from {args.test_dir}...")
    test_dataset = ExtractedFaceDataset(image_dir=args.test_dir, detector=detector)

    if len(test_dataset) == 0:
        logging.error("No images found in test directory.")
        return

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Model and Load Custom Weights
    logging.info("Loading fine-tuned model architecture...")
    model = setup_finetuning_model(
        weights_path=Path("models/vgg_face_dag.pth"),
        num_classes=len(classes),
        device=device,
    )

    # Load our trained parameters
    model.load_state_dict(
        torch.load(args.weights, map_location=device, weights_only=True)
    )
    model.eval()

    # 5. Run Evaluation
    logging.info("Starting evaluation on Test Set...")
    correct = 0
    total = 0
    unknown_imposters_forced_in = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)

            # Map the test label strings to the correct Train integer index
            # If a person in the test set wasn't in the train set (an Imposter),
            # we track them. Softmax will force them into a known class anyway.
            mapped_labels = []
            for lbl_idx in labels:
                person_name = test_dataset.classes[lbl_idx.item()]
                if person_name in class_to_idx:
                    mapped_labels.append(class_to_idx[person_name])
                else:
                    # This is a stranger!
                    mapped_labels.append(-1)
                    unknown_imposters_forced_in += 1

            mapped_labels = torch.tensor(mapped_labels).to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Only calculate standard accuracy for known members
            valid_mask = mapped_labels != -1
            if valid_mask.any():
                total += valid_mask.sum().item()
                correct += (
                    (predicted[valid_mask] == mapped_labels[valid_mask]).sum().item()
                )

    if total > 0:
        test_acc = 100.0 * correct / total
        logging.info(f"==================================================")
        logging.info(f"🎯 FINAL TEST ACCURACY (Known Members): {test_acc:.2f}%")
        logging.info(f"==================================================")

    if unknown_imposters_forced_in > 0:
        logging.warning(
            f"🚨 DETECTED {unknown_imposters_forced_in} STRANGER FRAMES IN TEST SET!"
        )
        logging.warning(
            "Because this is a Closed-Set model, 100% of these strangers were FALSELY ACCEPTED as team members."
        )
        logging.warning(
            "This proves the mathematical necessity of Approach 1 (Open-Set Metric Learning)."
        )


if __name__ == "__main__":
    main()
