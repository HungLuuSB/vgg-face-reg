"""
Fine-tuning module for the vgg-face-reg project (Ablation Study).

This script implements Approach 2: Fine-Tuning. It loads the VGGFace weights,
freezes the early convolutional layers (conv1 to conv4), unfreezes conv5,
splits the dataset into Training (80%) and Validation (20%), and trains a new
Classification Head. Metrics are logged and exported to a CSV file.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from core.embedder import VGGFaceEmbedder
from core.detector import FaceDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ExtractedFaceDataset(Dataset):
    """
    Custom Dataset to load raw frames, detect faces via MTCNN, and return tensors.
    """

    def __init__(self, image_dir: Path, detector: FaceDetector):
        self.image_paths = list(image_dir.glob("*.jpg"))
        self.detector = detector

        # Extract unique classes from filenames (e.g., 'Hung_0001.jpg' -> 'Hung')
        raw_labels = [path.stem.split("_")[0] for path in self.image_paths]
        self.classes = sorted(list(set(raw_labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Valid data ledger
        self.valid_data: List[dict] = []
        logging.info(f"Scanning {len(self.image_paths)} images to build dataset...")

        for path in self.image_paths:
            label_str = path.stem.split("_")[0]
            self.valid_data.append(
                {"path": path, "label": self.class_to_idx[label_str]}
            )

        logging.info(
            f"Dataset built. Found {len(self.classes)} classes: {self.classes}"
        )

    def __len__(self) -> int:
        return len(self.valid_data)

    def __getitem__(self, idx: int) -> tuple:
        item = self.valid_data[idx]
        img_path = item["path"]
        label = item["label"]

        # Process through MTCNN (align, crop, normalize)
        face_tensor = self.detector.process_frame(img_path)

        # Fallback if MTCNN fails on a specific frame during training
        if face_tensor is None:
            face_tensor = torch.zeros((3, 224, 224))

        return face_tensor, label


def setup_finetuning_model(
    weights_path: Path, num_classes: int, device: torch.device
) -> nn.Module:
    """
    Initializes VGG-16, loads VGGFace weights, applies freezing logic, and sets new head.
    """
    embedder = VGGFaceEmbedder(weights_path=weights_path, device=device)
    model = embedder.model

    # Freezing Strategy: Freeze conv1 through conv4. Unfreeze conv5.
    for name, param in model.features.named_parameters():
        layer_index = int(name.split(".")[0])
        if layer_index < 24:
            param.requires_grad = False  # Freeze
        else:
            param.requires_grad = True  # Unfreeze conv5 block

    # Replace the Classification Head with Global Average Pooling to save VRAM
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),  # Output matches exactly 5 team members
    )

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VGG-Face (Approach 2)")
    parser.add_argument("-i", "--images_dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "-w", "--weights", type=Path, default=Path("models/vgg_face_dag.pth")
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=Path("data/evaluation/finetune_metrics.csv"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Prepare Dataset and Train/Val Split
    detector = FaceDetector(image_size=224, device=device)
    full_dataset = ExtractedFaceDataset(image_dir=args.images_dir, detector=detector)

    if len(full_dataset) == 0:
        logging.error("No valid images found for training.")
        return

    # Calculate split sizes (80% Train, 20% Validation)
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # ---- GỌI HÀM IN THỐNG KÊ TẠI ĐÂY ----
    print_dataset_statistics(full_dataset, train_dataset, val_dataset)
    # ---------------------------------------

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 2. Prepare Model
    num_classes = len(full_dataset.classes)
    model = setup_finetuning_model(args.weights, num_classes, device)

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # 4. Training Loop with Validation and Metric Tracking
    logging.info(
        f"Starting Fine-Tuning for {args.epochs} epochs with Batch Size {args.batch_size}..."
    )

    metrics_history = []

    for epoch in range(args.epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"
        )
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Track Training Metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                loss=loss.item(), acc=100.0 * train_correct / train_total
            )

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Valid]", leave=False
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track Validation Metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        logging.info(
            f"Epoch {epoch + 1} Summary | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )

        # Store metrics for CSV export
        metrics_history.append(
            {
                "Epoch": epoch + 1,
                "Train_Loss": round(epoch_train_loss, 4),
                "Train_Acc": round(epoch_train_acc, 2),
                "Val_Loss": round(epoch_val_loss, 4),
                "Val_Acc": round(epoch_val_acc, 2),
            }
        )

    # 5. Export Metrics to CSV
    try:
        with open(args.output_csv, mode="w", newline="") as csv_file:
            fieldnames = ["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for row in metrics_history:
                writer.writerow(row)
        logging.info(f"Metrics successfully saved to {args.output_csv}")
    except Exception as e:
        logging.error(f"Failed to write metrics to CSV: {e}")

    # Optional: Save the fine-tuned weights
    torch.save(model.state_dict(), "models/vgg_face_finetuned.pth")
    logging.info(
        "Fine-Tuning complete. Check the CSV file to plot your learning curves."
    )


def print_dataset_statistics(
    full_dataset: ExtractedFaceDataset, train_dataset, val_dataset
):
    """
    Tính toán và in ra thống kê chi tiết của tập dữ liệu dưới dạng cấu trúc cây.
    """
    # Khởi tạo bộ đếm
    train_counts = {c: 0 for c in full_dataset.classes}
    val_counts = {c: 0 for c in full_dataset.classes}

    # Đếm số lượng ảnh từng class trong tập Train
    for idx in train_dataset.indices:
        label_idx = full_dataset.valid_data[idx]["label"]
        class_name = full_dataset.classes[label_idx]
        train_counts[class_name] += 1

    # Đếm số lượng ảnh từng class trong tập Valid
    for idx in val_dataset.indices:
        label_idx = full_dataset.valid_data[idx]["label"]
        class_name = full_dataset.classes[label_idx]
        val_counts[class_name] += 1

    # In ra dạng cấu trúc cây (Tree structure)
    print("\n" + "=" * 50)
    print("📊 DATASET SPLIT SUMMARY (Ablation Study)")
    print("=" * 50)
    print(f"Total Processed Images: {len(full_dataset)}")

    # Nhánh Training
    print(f"├── [Training Set] : {len(train_dataset)} images (80%)")
    for i, c in enumerate(full_dataset.classes):
        prefix = "│   ├──" if i < len(full_dataset.classes) - 1 else "│   └──"
        print(f"{prefix} {c}: {train_counts[c]} images")

    # Nhánh Validation
    print(f"└── [Validation Set] : {len(val_dataset)} images (20%)")
    for i, c in enumerate(full_dataset.classes):
        prefix = "    ├──" if i < len(full_dataset.classes) - 1 else "    └──"
        print(f"{prefix} {c}: {val_counts[c]} images")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
