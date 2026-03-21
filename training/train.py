"""Training pipeline for fighting detection model.

Supports:
    - Mixed precision training (FP16) for memory efficiency
    - Cosine annealing learning rate schedule with warmup
    - Backbone freezing for transfer learning
    - Gradient clipping
    - Label smoothing
    - Checkpoint saving (best and periodic) organized by model and run
    - Training/validation loss and accuracy tracking
    - Per-epoch confusion matrices and metadata tracking

Usage:
    python -m training.train --config configs/default.yaml
"""

import argparse
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FightDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.factory import create_model
from utils.config import load_config
from utils.logger import setup_logger
from utils.metrics import MetricsTracker, compute_classification_metrics
from utils.visualization import plot_training_curves, plot_confusion_matrix

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """Generate a unique run ID combining timestamp and UUID.

    Returns:
        String in format: run_YYYYMMDD_HHMMSS_<uuid_first_8_chars>
        Example: run_20260315_150230_a1b2c3d4
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_short = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{uuid_short}"


def build_optimizer(model, config):
    """Build optimizer with optional per-layer learning rate scaling.

    Args:
        model: The model to optimize.
        config: Training configuration.

    Returns:
        Configured optimizer.
    """
    lr = config.training.learning_rate
    weight_decay = config.training.weight_decay

    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(backbone_lr_scale=0.1)
        for group in param_groups:
            group["lr"] = lr * group.pop("lr_scale")
            group["weight_decay"] = weight_decay
    else:
        param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]

    optimizer_name = config.training.get("optimizer", "sgd")
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=config.training.get("momentum", 0.9),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch: int):
    """Build learning rate scheduler.

    Args:
        optimizer: The optimizer.
        config: Training configuration.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        Learning rate scheduler.
    """
    num_epochs = config.training.num_epochs
    warmup_epochs = config.training.get("warmup_epochs", 0)
    scheduler_name = config.training.get("lr_scheduler", "cosine")

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6,
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
        )
    else:
        scheduler = None

    return scheduler


def warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    """Apply linear warmup to learning rate.

    Args:
        optimizer: The optimizer.
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of warmup epochs.
        base_lr: Target learning rate after warmup.
    """
    if warmup_epochs <= 0:
        return
    lr = base_lr * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale_factor", 1.0)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    grad_clip_norm: float = 0.0,
    grad_accumulation_steps: int = 1,
) -> tuple:
    """Train for one epoch with gradient accumulation support.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        scaler: Gradient scaler for mixed precision.
        use_amp: Whether to use automatic mixed precision.
        grad_clip_norm: Max norm for gradient clipping. 0 to disable.
        grad_accumulation_steps: Steps to accumulate gradients. Default: 1 (no accumulation).

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    accumulation_counter = 0

    for clips, labels in tqdm(dataloader, desc="Training", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(clips)
            loss = criterion(outputs, labels)
            # Scale loss by accumulation steps to maintain consistent gradient magnitude
            loss = loss / grad_accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulation_counter += 1

        # Update weights after accumulation
        if accumulation_counter % grad_accumulation_steps == 0:
            if use_amp:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            optimizer.zero_grad()
            accumulation_counter = 0

        running_loss += loss.item() * grad_accumulation_steps * clips.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple:
    """Validate the model.

    Args:
        model: The model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Tuple of (average_loss, accuracy, all_predictions, all_labels).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for clips, labels in tqdm(dataloader, desc="Validation", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(clips)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * clips.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: str,
):
    """Save a training checkpoint.

    Args:
        model: The model.
        optimizer: The optimizer.
        epoch: Current epoch.
        metrics: Current metrics dictionary.
        save_path: Path to save the checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, save_path)
    logger.info("Checkpoint saved: %s", save_path)


def train(config, run_id: str = None):
    """Main training function.

    Args:
        config: Configuration object.
        run_id: Unique run identifier. Generated if not provided.
    """
    # Generate run_id if not provided
    if run_id is None:
        run_id = generate_run_id()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Organize checkpoints and logs by model name and run_id
    model_name = config.model.name
    checkpoint_base = Path(config.training.checkpoint_dir)
    checkpoint_dir = checkpoint_base / model_name / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_base = Path(config.training.get("log_dir", "logs"))
    log_dir = log_base / model_name / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run ID: %s", run_id)
    logger.info("Checkpoint directory: %s", checkpoint_dir)
    logger.info("Log directory: %s", log_dir)

    # Build transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    # Build datasets
    classes = config.data.get("classes", ["normal", "fight"])
    if isinstance(classes, list) and len(classes) > 0 and hasattr(classes[0], "to_dict"):
        classes = [c.to_dict() if hasattr(c, "to_dict") else str(c) for c in classes]

    train_dataset = FightDataset(
        data_dir=config.data.train_dir,
        clip_length=config.data.clip_length,
        frame_stride=config.data.frame_stride,
        transform=train_transform,
        classes=classes,
        sampling_mode="random",
    )

    val_dataset = FightDataset(
        data_dir=config.data.val_dir,
        clip_length=config.data.clip_length,
        frame_stride=config.data.frame_stride,
        transform=val_transform,
        classes=classes,
        sampling_mode="uniform",
    )

    # Validate datasets before creating loaders
    if len(train_dataset) == 0:
        logger.error(
            "No training samples found in %s. "
            "Please add video files organized as: "
            "<train_dir>/fight/*.avi and <train_dir>/normal/*.avi",
            config.data.train_dir,
        )
        return

    if len(val_dataset) == 0:
        logger.warning(
            "No validation samples found in %s. "
            "Training will proceed without validation.",
            config.data.val_dir,
        )

    # Build data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    # Build model
    model = create_model(config)
    model = model.to(device)

    # Loss function with optional label smoothing
    label_smoothing = config.training.get("label_smoothing", 0.0)
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # Mixed precision
    use_amp = config.training.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger.info("Mixed precision training: %s", use_amp)

    # Gradient clipping
    grad_clip_norm = config.training.get("gradient_clip_norm", 0.0)

    # Gradient accumulation
    grad_accumulation_steps = config.training.get("gradient_accumulation_steps", 1)
    if grad_accumulation_steps > 1:
        logger.info(
            "Gradient accumulation enabled: %d steps (simulates batch_size=%d)",
            grad_accumulation_steps,
            config.training.batch_size * grad_accumulation_steps,
        )

    # Backbone freezing
    freeze_epochs = config.training.get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        model.freeze_backbone()
        logger.info("Backbone frozen for first %d epochs", freeze_epochs)

    # Metrics tracker
    tracker = MetricsTracker()

    # Training loop
    num_epochs = config.training.num_epochs
    warmup_epochs = config.training.get("warmup_epochs", 0)

    logger.info("Starting training for %d epochs", num_epochs)
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Unfreeze backbone after freeze period
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.unfreeze_backbone()
            logger.info("Backbone unfrozen at epoch %d", epoch + 1)

        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_lr(optimizer, epoch, warmup_epochs, config.training.learning_rate)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, use_amp, grad_clip_norm, grad_accumulation_steps,
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        tracker.update_train(train_loss, train_acc, current_lr)

        # Validate
        val_interval = config.training.get("val_interval", 1)
        run_val = val_loader is not None and (epoch + 1) % val_interval == 0
        if run_val:
            val_loss, val_acc, val_preds, val_labels = validate(
                model, val_loader, criterion, device, use_amp,
            )
            is_best = tracker.update_val(val_loss, val_acc)

            if is_best:
                save_checkpoint(
                    model, optimizer, epoch,
                    {"val_loss": val_loss, "val_acc": val_acc},
                    checkpoint_dir / "best_model.pth",
                )

            epoch_time = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d - train_loss: %.4f, train_acc: %.4f, "
                "val_loss: %.4f, val_acc: %.4f, lr: %.6f, time: %.1fs%s",
                epoch + 1, num_epochs,
                train_loss, train_acc, val_loss, val_acc,
                current_lr, epoch_time,
                " [BEST]" if is_best else "",
            )

            # Compute and save detailed metrics at save intervals and final epoch
            save_interval = config.training.get("save_interval", 5)
            is_save_epoch = (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1

            if is_save_epoch:
                class_metrics = compute_classification_metrics(
                    val_labels, val_preds, class_names=classes,
                )
                logger.info(
                    "Epoch %d metrics - Accuracy: %.4f, Precision: %.4f, "
                    "Recall: %.4f, F1: %.4f",
                    epoch + 1,
                    class_metrics["accuracy"],
                    class_metrics["precision"],
                    class_metrics["recall"],
                    class_metrics["f1_score"],
                )
                # Save per-epoch confusion matrix
                cm_filename = f"confusion_matrix_epoch_{epoch + 1}.png"
                plot_confusion_matrix(
                    np.array(class_metrics["confusion_matrix"]),
                    classes,
                    str(log_dir / cm_filename),
                )
                logger.info("Saved confusion matrix: %s", cm_filename)
        else:
            epoch_time = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d - train_loss: %.4f, train_acc: %.4f, "
                "lr: %.6f, time: %.1fs",
                epoch + 1, num_epochs,
                train_loss, train_acc, current_lr, epoch_time,
            )

        # Step scheduler after warmup
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        # Periodic checkpoint
        save_interval = config.training.get("save_interval", 5)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {"train_loss": train_loss, "train_acc": train_acc},
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
            )

    # Save final model (also as best_model if no validation was performed)
    save_checkpoint(
        model, optimizer, num_epochs - 1,
        tracker.get_summary(),
        checkpoint_dir / "final_model.pth",
    )

    if val_loader is None:
        save_checkpoint(
            model, optimizer, num_epochs - 1,
            {"train_loss": tracker.train_losses[-1], "train_acc": tracker.train_accuracies[-1]},
            checkpoint_dir / "best_model.pth",
        )
        logger.info("No validation data was provided. Final model saved as best_model.pth")

    # Save metrics and plots
    tracker.save(str(log_dir / "training_metrics.json"))
    plot_training_curves(tracker, str(log_dir))

    # Save metadata about the training run
    metadata = {
        "run_id": run_id,
        "model_name": config.model.name,
        "model_source": config.model.get("source", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "training_config": {
            "batch_size": config.training.batch_size,
            "num_epochs": config.training.num_epochs,
            "learning_rate": config.training.learning_rate,
            "optimizer": config.training.get("optimizer", "sgd"),
            "warmup_epochs": config.training.get("warmup_epochs", 0),
            "freeze_backbone_epochs": config.training.get("freeze_backbone_epochs", 0),
        },
        "model_config": {
            "clip_length": config.model.clip_length,
            "spatial_size": config.model.spatial_size,
            "frame_stride": config.model.get("frame_stride", 2),
            "pretrained": config.model.pretrained,
            "num_classes": config.model.num_classes,
        },
        "best_epoch": tracker.best_epoch + 1 if tracker.best_epoch >= 0 else None,
        "best_val_accuracy": float(tracker.best_val_accuracy) if tracker.best_val_accuracy > 0 else None,
        "best_val_loss": float(tracker.best_val_loss) if tracker.best_val_loss > 0 else None,
        "final_metrics": {
            "train_loss": float(tracker.train_losses[-1]) if tracker.train_losses else None,
            "train_accuracy": float(tracker.train_accuracies[-1]) if tracker.train_accuracies else None,
            "val_loss": float(tracker.val_losses[-1]) if tracker.val_losses else None,
            "val_accuracy": float(tracker.val_accuracies[-1]) if tracker.val_accuracies else None,
        },
    }
    metadata_path = log_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: %s", metadata_path)

    if tracker.best_val_accuracy > 0:
        logger.info("Training complete. Best val accuracy: %.4f at epoch %d",
                    tracker.best_val_accuracy, tracker.best_epoch + 1)
    else:
        logger.info("Training complete. Final train accuracy: %.4f",
                    tracker.train_accuracies[-1] if tracker.train_accuracies else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train fighting detection model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Generate unique run ID
    run_id = generate_run_id()

    # Setup logging
    log_base = Path(config.training.get("log_dir", "logs"))
    model_name = config.model.name
    log_dir = log_base / model_name / run_id
    setup_logger("root", log_dir=str(log_dir))
    setup_logger(__name__, log_dir=str(log_dir))

    logger.info("=" * 80)
    logger.info("Starting training run: %s", run_id)
    logger.info("Model: %s", model_name)
    logger.info("=" * 80)

    train(config, run_id=run_id)


if __name__ == "__main__":
    main()
