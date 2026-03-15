"""Training metrics tracking and computation."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class MetricsTracker:
    """Tracks training and validation metrics across epochs."""

    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        self.best_val_loss: float = float("inf")
        self.best_val_accuracy: float = 0.0
        self.best_epoch: int = 0

    def update_train(self, loss: float, accuracy: float, lr: float):
        """Record training metrics for one epoch."""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(lr)

    def update_val(self, loss: float, accuracy: float) -> bool:
        """Record validation metrics for one epoch.

        Returns:
            True if this is the best validation loss so far.
        """
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)

        is_best = loss < self.best_val_loss
        if is_best:
            self.best_val_loss = loss
            self.best_val_accuracy = accuracy
            self.best_epoch = len(self.val_losses) - 1

        return is_best

    def get_summary(self) -> Dict:
        """Get a summary of all tracked metrics."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
        }

    def save(self, save_path: str):
        """Save metrics to a JSON file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

    @classmethod
    def load(cls, load_path: str) -> "MetricsTracker":
        """Load metrics from a JSON file."""
        with open(load_path, "r") as f:
            data = json.load(f)
        tracker = cls()
        tracker.train_losses = data["train_losses"]
        tracker.val_losses = data["val_losses"]
        tracker.train_accuracies = data["train_accuracies"]
        tracker.val_accuracies = data["val_accuracies"]
        tracker.learning_rates = data["learning_rates"]
        tracker.best_val_loss = data["best_val_loss"]
        tracker.best_val_accuracy = data["best_val_accuracy"]
        tracker.best_epoch = data["best_epoch"]
        return tracker


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class names.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    if class_names is None:
        class_names = [str(i) for i in range(max(y_true.max(), y_pred.max()) + 1)]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": class_names,
    }
