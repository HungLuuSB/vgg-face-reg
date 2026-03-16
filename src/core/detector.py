"""
Face detection and alignment module for the vgg-face-reg project.

This module utilizes the MTCNN (Multitask Cascaded Convolutional Networks)
architecture to locate faces, extract 5-point landmarks, and apply an affine
transformation to return a geometrically normalized PyTorch tensor.
"""

import logging
from typing import Optional, Tuple, Union

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class FaceDetector:
    """
    A wrapper class for the facenet-pytorch MTCNN implementation.

    Attributes:
        device (torch.device): The hardware device (CPU or CUDA) for tensor operations.
        mtcnn (MTCNN): The instantiated MTCNN model.
    """

    def __init__(
        self,
        image_size: int = 224,
        margin: int = 0,
        min_face_size: int = 40,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes the FaceDetector with specific geometric constraints.

        Args:
            image_size (int): The width and height of the square output tensor.
            margin (int): Padding to add to the bounding box before cropping.
            min_face_size (int): Minimum pixel height/width of a face to be detected.
            device (Optional[torch.device]): Target compute device. Defaults to
                CUDA if available, otherwise CPU.
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logging.info(f"Initializing MTCNN FaceDetector on device: {self.device}")

        # Instantiate the MTCNN model
        # keep_all=False ensures we only extract the highest-probability face per frame
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            keep_all=False,
            device=self.device,
            post_process=True,  # Normalizes pixel values to [-1, 1] for neural network input
        )

    def process_frame(
        self, image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Optional[torch.Tensor]:
        """
        Detects, aligns, and crops a face from an input image.

        Args:
            image_input (Union[str, Path, np.ndarray, Image.Image]): The source image.
                Can be a file path, an OpenCV numpy array (BGR), or a PIL Image.

        Returns:
            Optional[torch.Tensor]: A normalized PyTorch tensor of shape
            (3, image_size, image_size) representing the aligned face.
            Returns None if no face is detected.
        """
        # Standardize input to a PIL Image (RGB) as required by facenet-pytorch
        img_pil: Image.Image

        if isinstance(image_input, (str, Path)):
            img_pil = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Convert OpenCV BGR array to RGB PIL Image
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        elif isinstance(image_input, Image.Image):
            img_pil = image_input
        else:
            raise TypeError("Unsupported image_input type.")

        # Detect and align the face.
        # mtcnn() returns the normalized cropped tensor directly.
        face_tensor: Optional[torch.Tensor] = self.mtcnn(img_pil)

        if face_tensor is None:
            logging.debug("No face detected in the provided frame.")
            return None

        return face_tensor
