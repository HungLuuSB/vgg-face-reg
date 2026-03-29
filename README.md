# VGG-Face Real-Time Open-Set Verification (vgg-face-reg)

A real-time, deep metric learning facial verification system built with PyTorch, OpenCV, and MTCNN.

Unlike traditional closed-set classifiers (Softmax) that are forced to guess a known identity, this system utilizes a stripped VGG-16 architecture to project faces into a continuous 4096-dimensional embedding space. By leveraging Cosine Similarity and $k$-Nearest Neighbors, it performs **Open-Set Verification**, reliably identifying registered users while mathematically rejecting strangers as "Unknown."

## Features

* **MTCNN Alignment:** 2D geometric landmark detection and affine transformations for strict $224 \times 224$ facial alignment.
* **Deep Metric Learning:** Pre-trained VGG-16 feature extraction without classification head constraints.
* **Vectorized Matching:** Optimized matrix multiplication for real-time $k$-NN Cosine Similarity against the reference gallery.
* **Scientific Evaluation Pipeline:** Automated pair generation via $R_{yaw}$ and $R_{pitch}$ geometric ratios to calculate the Equal Error Rate (EER) and ROC curve.

## Project Structure

```text
vgg-face-reg/
├── data/                  # Ignored by git
│   ├── raw/               # Raw .mp4 video files
│   ├── processed/         # Extracted .jpg frames for the gallery
│   ├── evaluation/        # Extracted .jpg frames for ROC testing
│   └── gallery/           # Compiled reference.pt database
├── models/                # Ignored by git (Place vgg_face_dag.pth here)
├── notebooks/
│   └── evaluate.ipynb     # Jupyter notebook for EER and ROC curve generation
├── src/
│   ├── core/
│   │   ├── detector.py    # MTCNN implementation
│   │   ├── embedder.py    # VGG-16 architecture mapping
│   │   └── matcher.py     # Vectorized Cosine Similarity & k-NN
│   ├── utils/
│   │   ├── video.py       # Dynamic FPS frame extraction
│   │   ├── metrics.py     # Automated Genuine/Imposter pair generation
│   │   └── build_gallery.py # Compiles reference embeddings to .pt
│   └── cli/
│       └── main.py        # Real-time OpenCV inference entry point
├── requirements.txt
└── README.md
```

## User Guide

### 1. Environment Setup

Initialize your virtual environment and install the strict dependency matrix.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Download Weights:** Download the ported PyTorch VGGFace weights (`vgg_face_dag.pth`) and place them in the `models/` directory.

### 2. Pipeline Execution

**Phase 1: Data Acquisition**
Place your raw `.mp4` video files into `data/raw/` (ensure files are uniquely named per identity, e.g., `Alice.mp4`). Extract temporally spaced frames using the dynamic extraction tool:

```bash
PYTHONPATH=src python src/utils/video.py -i data/raw -o data/processed --fps 9.0
PYTHONPATH=src python src/utils/video.py -i data/test -o data/evaluation --fps 9.0
```

**Phase 2: Building the Reference Gallery**
Compile the extracted frames into an optimized, serialized PyTorch tensor dictionary (`reference.pt`):

```bash
PYTHONPATH=src python src/utils/build_gallery.py \
    --images_dir data/processed \
    --weights models/vgg_face_dag.pth \
    --output data/gallery/reference.pt
```

**Phase 3: Scientific Evaluation (Finding the Threshold)**
To calculate your system's exact Equal Error Rate (EER) threshold based on your dataset:

1. Extract testing videos to `data/evaluation/`.
2. Generate the Genuine/Imposter pair ledger:

```bash
PYTHONPATH=src python src/utils/metrics.py -i data/evaluation -o data/evaluation/pairs_ledger.csv --genuine 2000 --imposter 2000
```

1. Run `notebooks/evaluate.ipynb` to output the optimal EER threshold ($\tau$).

**Phase 4: Real-Time Inference**
Launch the live camera feed. Adjust the --threshold flag based on your EER calculation, or increase it (e.g., 0.93) to strictly lock out Hard Imposters (siblings/similar demographics).

```bash
PYTHONPATH=src python src/cli/main.py --approach zeroshot --weights models/vgg_face_dag.pth --gallery data/gallery/reference.pt --threshold 0.85
PYTHONPATH=src python src/cli/main.py --approach finetuned --weights models/vgg_face_finetuned.pth --threshold 0.90
```

**For the 2nd approach:**

```bash
PYTHONPATH=src python src/utils/train_finetune.py --epochs 10 --batch_size 4 --lr 0.00001
PYTHONPATH=src python src/utils/test_finetune.py
PYTHONPATH=src python src/utils/plot_metrics.py
```
