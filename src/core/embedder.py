"""
Feature extraction module for the vgg-face-reg project.

This module initializes a VGG-16 architecture, loads pre-trained VGGFace weights,
and strips the final classification layer to output a 4096-dimensional embedding
vector for metric learning.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class VGGFaceEmbedder:
    """
    A wrapper class for the VGG-16 network adapted for facial feature extraction.

    Attributes:
        device (torch.device): The hardware device for tensor operations.
        model (nn.Module): The modified VGG-16 network.
    """

    def __init__(
        self, weights_path: Path, device: Optional[torch.device] = None
    ) -> None:
        """
        Initializes the VGG-16 model for embedding extraction.

        Args:
            weights_path (Path): Path to the VGGFace .pth weights file.
            device (Optional[torch.device]): Target compute device.
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logging.info(f"Initializing VGGFace on device: {self.device}")

        # 1. Load the base VGG-16 architecture
        self.model = vgg16(weights=None)

        # 2. Modify the classifier to match the original VGGFace architecture
        self.model.classifier[6] = nn.Linear(4096, 2622)

        # 3. Load the pre-trained VGGFace weights into memory
        if not weights_path.exists():
            raise FileNotFoundError(f"VGGFace weights not found at: {weights_path}")

        raw_state_dict = torch.load(
            str(weights_path), map_location=self.device, weights_only=True
        )

        # 4. Dictionary Key Translation (Caffe/DAG format -> torchvision format)
        key_mapping = {
            "conv1_1.weight": "features.0.weight",
            "conv1_1.bias": "features.0.bias",
            "conv1_2.weight": "features.2.weight",
            "conv1_2.bias": "features.2.bias",
            "conv2_1.weight": "features.5.weight",
            "conv2_1.bias": "features.5.bias",
            "conv2_2.weight": "features.7.weight",
            "conv2_2.bias": "features.7.bias",
            "conv3_1.weight": "features.10.weight",
            "conv3_1.bias": "features.10.bias",
            "conv3_2.weight": "features.12.weight",
            "conv3_2.bias": "features.12.bias",
            "conv3_3.weight": "features.14.weight",
            "conv3_3.bias": "features.14.bias",
            "conv4_1.weight": "features.17.weight",
            "conv4_1.bias": "features.17.bias",
            "conv4_2.weight": "features.19.weight",
            "conv4_2.bias": "features.19.bias",
            "conv4_3.weight": "features.21.weight",
            "conv4_3.bias": "features.21.bias",
            "conv5_1.weight": "features.24.weight",
            "conv5_1.bias": "features.24.bias",
            "conv5_2.weight": "features.26.weight",
            "conv5_2.bias": "features.26.bias",
            "conv5_3.weight": "features.28.weight",
            "conv5_3.bias": "features.28.bias",
            "fc6.weight": "classifier.0.weight",
            "fc6.bias": "classifier.0.bias",
            "fc7.weight": "classifier.3.weight",
            "fc7.bias": "classifier.3.bias",
            "fc8.weight": "classifier.6.weight",
            "fc8.bias": "classifier.6.bias",
        }

        translated_dict = {}
        for old_key, tensor in raw_state_dict.items():
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
                translated_dict[new_key] = tensor

        # Load the translated dictionary
        self.model.load_state_dict(translated_dict)

        # 5. Strip the classification head for Metric Learning
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-3]
        )

        # 6. Prepare model for inference
        self.model = self.model.to(self.device)
        self.model.eval()

        logging.info("VGGFace Embedder initialized and classification head stripped.")

    @torch.no_grad()
    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        Passes an aligned face tensor through the network to extract the embedding.

        Args:
            face_tensor (torch.Tensor): A normalized face tensor of shape
                (3, 224, 224) or a batch (B, 3, 224, 224).

        Returns:
            torch.Tensor: A 4096-dimensional feature vector, moved to the CPU
                for downstream distance calculations.
        """
        # Ensure the tensor has a batch dimension (B, C, H, W)
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)

        face_tensor = face_tensor.to(self.device)

        # Extract features (Shape: [B, 4096])
        embedding: torch.Tensor = self.model(face_tensor)

        # Normalize the embedding using L2 normalization
        # This projects the vector onto a unit hypersphere, which is mathematically
        # required before calculating Cosine Similarity.
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu()
