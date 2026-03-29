"""
Plotting utility for the vgg-face-reg project.

Reads the finetune_metrics.csv file and generates a high-resolution
learning curve chart (Loss and Accuracy) for the academic report.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def plot_learning_curves(
    csv_path: Path = Path("data/evaluation/finetune_metrics.csv"),
    output_path: Path = Path("data/evaluation/learning_curves.png"),
) -> None:

    if not csv_path.exists():
        logging.error(
            f"Metrics file not found at {csv_path}. Please run train_finetune.py first."
        )
        return

    # 1. Đọc dữ liệu từ file CSV
    logging.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Cài đặt font chữ chuẩn học thuật
    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    # 2. Tạo khung biểu đồ với 2 đồ thị con (1 hàng, 2 cột)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- ĐỒ THỊ 1: LOSS CURVE ---
    ax1.plot(
        df["Epoch"],
        df["Train_Loss"],
        label="Train Loss",
        color="#1f77b4",
        linewidth=2,
        marker="o",
    )
    ax1.plot(
        df["Epoch"],
        df["Val_Loss"],
        label="Validation Loss",
        color="#d62728",
        linewidth=2,
        marker="s",
    )

    ax1.set_title("Đồ thị suy giảm Hàm mất mát (Loss Curve)", fontsize=14, pad=15)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_xticks(df["Epoch"])  # Hiển thị chính xác từng số Epoch ở trục X
    ax1.legend(loc="upper right", frameon=True, shadow=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- ĐỒ THỊ 2: ACCURACY CURVE ---
    ax1.plot()  # Reset background
    ax2.plot(
        df["Epoch"],
        df["Train_Acc"],
        label="Train Accuracy",
        color="#1f77b4",
        linewidth=2,
        marker="o",
    )
    ax2.plot(
        df["Epoch"],
        df["Val_Acc"],
        label="Validation Accuracy",
        color="#2ca02c",
        linewidth=2,
        marker="^",
    )

    ax2.set_title("Đồ thị Độ chính xác (Accuracy Curve)", fontsize=14, pad=15)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_xticks(df["Epoch"])
    ax2.legend(loc="lower right", frameon=True, shadow=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Căn chỉnh và Lưu file
    plt.tight_layout()
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight"
    )  # dpi=300 đảm bảo ảnh in ra Word cực nét
    logging.info(f"Chart successfully saved at: {output_path}")

    # Hiển thị lên màn hình
    plt.show()


if __name__ == "__main__":
    plot_learning_curves()
