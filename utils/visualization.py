"""Visualization utilities for training curves and results."""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import MetricsTracker


def plot_training_curves(
    tracker: MetricsTracker,
    save_dir: str,
    show: bool = False,
):
    """Plot and save training/validation loss and accuracy curves.

    Args:
        tracker: MetricsTracker instance with recorded metrics.
        save_dir: Directory to save the plot images.
        show: Whether to display the plots interactively.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(tracker.train_losses) + 1))

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, tracker.train_losses, label="Train Loss", linewidth=2)
    if tracker.val_losses:
        val_epochs = list(range(1, len(tracker.val_losses) + 1))
        ax.plot(val_epochs, tracker.val_losses, label="Val Loss", linewidth=2)
        ax.axvline(
            x=tracker.best_epoch + 1,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Best (epoch {tracker.best_epoch + 1})",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curves.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, tracker.train_accuracies, label="Train Accuracy", linewidth=2)
    if tracker.val_accuracies:
        val_epochs = list(range(1, len(tracker.val_accuracies) + 1))
        ax.plot(val_epochs, tracker.val_accuracies, label="Val Accuracy", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_curves.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # Learning rate schedule
    if tracker.learning_rates:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, tracker.learning_rates, linewidth=2, color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "lr_schedule.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = True,
    show: bool = False,
):
    """Plot and save a confusion matrix.

    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        save_path: Path to save the plot.
        normalize: Whether to normalize the matrix.
        show: Whether to display the plot interactively.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_display = cm_normalized
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_display.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = f"{cm_display[i, j]:.2f}" if normalize else f"{cm_display[i, j]}"
            ax.text(
                j, i, value,
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
