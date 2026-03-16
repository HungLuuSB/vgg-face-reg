"""
Identity matching module for the vgg-face-reg project.

This module loads a pre-computed gallery of facial embeddings and performs
vectorized Cosine Similarity and k-Nearest Neighbors (k-NN) classification
to identify a live face tensor or reject it as "Unknown".
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class FaceMatcher:
    """
    A vectorized k-NN classifier for open-set face recognition.

    Attributes:
        gallery_embeddings (torch.Tensor): An (N, 4096) tensor of normalized features.
        gallery_labels (List[str]): A list of length N containing the identity strings.
        threshold (float): The minimum Cosine Similarity required to be "Known".
        k (int): The number of nearest neighbors to evaluate.
    """

    def __init__(self, gallery_path: Path, threshold: float = 0.75, k: int = 5) -> None:
        """
        Initializes the Matcher by loading the reference gallery into memory.

        Args:
            gallery_path (Path): Path to a saved PyTorch dictionary containing
                'embeddings' (Tensor) and 'labels' (List[str]).
            threshold (float): The Cosine Similarity threshold (tau). Defaults to 0.75.
            k (int): Number of neighbors for majority voting. Defaults to 5.
        """
        if not gallery_path.exists():
            raise FileNotFoundError(f"Gallery file not found at: {gallery_path}")

        logging.info(f"Loading reference gallery from {gallery_path}...")

        # Load the gallery data into CPU memory (fast enough for inference)
        gallery_data: Dict[str, any] = torch.load(str(gallery_path), weights_only=False)  # type: ignore

        self.gallery_embeddings: torch.Tensor = gallery_data.get("embeddings")  # type: ignore
        self.gallery_labels: List[str] = gallery_data.get("labels")  # type: ignore
        self.threshold: float = threshold
        self.k: int = k

        if self.gallery_embeddings is None or self.gallery_labels is None:
            raise ValueError(
                "Gallery file must contain 'embeddings' and 'labels' keys."
            )

        if self.gallery_embeddings.size(0) != len(self.gallery_labels):
            raise ValueError("Mismatch between number of embeddings and labels.")

        logging.info(
            f"Gallery loaded: {self.gallery_embeddings.size(0)} reference frames."
        )

    def identify(self, live_embedding: torch.Tensor) -> Tuple[str, float]:
        """
        Identifies a live face embedding using k-NN and thresholding.

        Args:
            live_embedding (torch.Tensor): A normalized feature vector of shape
                (1, 4096) extracted from the live camera feed.

        Returns:
            Tuple[str, float]: The predicted identity (or "Unknown") and the
                average similarity score of the matching neighbors.
        """
        if live_embedding.dim() == 1:
            live_embedding = live_embedding.unsqueeze(0)

        # 1. Vectorized Cosine Similarity calculation
        similarities: torch.Tensor = torch.mm(
            live_embedding, self.gallery_embeddings.t()
        )
        similarities = similarities.squeeze(0)

        # --- Dynamic K ---
        # Ensure we don't ask for more neighbors than exist in the gallery
        gallery_size: int = self.gallery_embeddings.size(0)
        effective_k: int = min(self.k, gallery_size)

        # 2. Find the top-k highest similarity scores and their indices
        topk_scores, topk_indices = torch.topk(similarities, effective_k)

        # 3. Extract the labels of the top-k nearest neighbors
        nearest_labels: List[str] = [
            self.gallery_labels[i.item()] for i in topk_indices
        ]

        # 4. Perform Majority Voting
        label_counts = Counter(nearest_labels)
        best_label, majority_count = label_counts.most_common(1)[0]

        # 5. Calculate the average score for the majority class
        majority_scores = [
            topk_scores[idx].item()
            for idx, label in enumerate(nearest_labels)
            if label == best_label
        ]
        avg_majority_score: float = sum(majority_scores) / len(majority_scores)

        # 6. Apply the Open-Set Threshold (tau)
        if avg_majority_score >= self.threshold:
            return best_label, avg_majority_score
        else:
            return "Unknown", avg_majority_score
