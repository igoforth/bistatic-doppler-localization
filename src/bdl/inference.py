# -*- coding: utf-8 -*-
"""
Doppler inference utilities: visualization, analysis, and accuracy metrics.

@author: PaRa
@author: igoforth
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def loss_to_score(
    losses: np.ndarray, percentile_min: int = 5, percentile_max: int = 95
) -> np.ndarray:
    min_loss = np.percentile(losses, percentile_min)
    max_loss = np.percentile(losses, percentile_max)

    if max_loss == min_loss:
        scaled = np.zeros_like(losses)
    else:
        scaled = np.clip((losses - min_loss) / (max_loss - min_loss), 0, 1)

    return (1 - scaled) * 100


def calculate_accuracy(
    outputs: np.ndarray, targets: np.ndarray, error_threshold: float = 0.01
) -> float:
    num_samples = len(targets)
    accuracies = np.empty(num_samples)

    for i in range(num_samples):
        difference = np.abs(targets[i] - outputs[i])
        accurate_pixels = difference <= error_threshold
        accuracies[i] = np.mean(accurate_pixels) * 100

    overall = float(np.mean(accuracies))
    print(f"Total Average Accuracy across all samples: {overall:.2f}%")
    return overall


def create_analysis_image(
    outputs: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
    output_file: str = "doppler_analysis.png",
) -> None:
    mse_loss_heatmap = np.zeros((28, 28))
    sample_heatmap = np.zeros((28, 28))

    for i in tqdm(indices, desc="Calculating MSE loss heatmap"):
        pred = outputs[i]
        actual = targets[i]

        squared_errors = (pred - actual) ** 2
        mse_loss_heatmap += squared_errors

        target_pos = np.unravel_index(np.argmax(actual), actual.shape)
        sample_heatmap[target_pos] += 1

    mse_loss_heatmap = np.divide(
        mse_loss_heatmap,
        sample_heatmap,
        out=np.zeros_like(mse_loss_heatmap),
        where=sample_heatmap > 0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    im1 = ax1.imshow(mse_loss_heatmap, cmap="viridis")
    ax1.set_title("MSE Loss Heatmap")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(sample_heatmap, cmap="YlOrRd")
    ax2.set_title("Sample Count Heatmap")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Analysis saved as {output_file}")


def create_animation(
    outputs: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
    output_file: str = "doppler_animation.mp4",
    fps: int = 10,
    max_frames: int = 500,
) -> None:
    indices = indices[: min(len(indices), max_frames)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    init_frame = np.zeros((28, 28))
    pred_plot = ax1.imshow(init_frame, cmap="gray", vmin=0, vmax=1, interpolation="linear")
    ax1.set_title("Predicted Doppler Map")
    actual_plot = ax2.imshow(init_frame, cmap="gray", vmin=0, vmax=1, interpolation="linear")
    ax2.set_title("Actual Doppler Map")
    diff_plot = ax3.imshow(init_frame, cmap="bwr", vmin=-1, vmax=1, interpolation="linear")
    ax3.set_title("Difference (Actual - Predicted)")

    plt.colorbar(pred_plot, ax=ax1)
    plt.colorbar(actual_plot, ax=ax2)
    plt.colorbar(diff_plot, ax=ax3)

    pbar = tqdm(total=len(indices))

    def update(frame: int):
        i = indices[frame]
        pred_plot.set_data(outputs[i])
        actual_plot.set_data(targets[i])
        diff_plot.set_data(targets[i] - outputs[i])
        pbar.update(1)
        return pred_plot, actual_plot, diff_plot

    anim = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=1000 / fps, blit=True
    )
    anim.save(output_file, writer="ffmpeg", fps=fps)
    plt.close(fig)
    pbar.close()

    print(f"Animation saved as {output_file}")
