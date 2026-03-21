"""Tests for training pipeline.

Tests:
- 1-epoch training with different models
- Checkpoint saving
- Metrics tracking
- Loss computation
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.factory import create_model, list_available_models
from utils.config import load_config


@pytest.mark.slow
class TestTrainingPipeline:
    """Test model training pipeline."""

    @pytest.mark.parametrize("model_name", ["x3d_s", "slowfast"])
    def test_single_epoch_training(self, model_name, device):
        """Test training for a single epoch."""
        config_path = f"configs/{model_name}.yaml"
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            config = load_config("configs/default.yaml")
            config.model.name = model_name

        # Create model
        model = create_model(config)
        model = model.to(device)
        model.train()

        # Create dummy data
        X = torch.randn(8, 3, 16, 224, 224)  # 8 samples, 16 frames
        y = torch.randint(0, 2, (8,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=2)

        # Setup training components
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Training loop for one epoch
        total_loss = 0.0
        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        assert avg_loss > 0, "Training loss should be positive"
        assert not torch.isnan(
            torch.tensor(avg_loss)
        ), "Training loss should not be NaN"

    def test_checkpoint_saving(self, device):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config("configs/x3d_s.yaml")
            model = create_model(config)
            model = model.to(device)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

            # Prepare checkpoint
            checkpoint = {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": 0.5,
            }

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)
            assert checkpoint_path.exists(), "Checkpoint file not saved"

            # Load checkpoint
            loaded = torch.load(checkpoint_path, weights_only=True)
            assert "epoch" in loaded
            assert "model_state_dict" in loaded
            assert loaded["epoch"] == 0

    def test_metrics_computation(self):
        """Test metrics computation during training."""
        from utils.metrics import compute_classification_metrics

        # Create dummy predictions and labels
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 1, 1]

        metrics = compute_classification_metrics(
            y_true, y_pred, class_names=["normal", "fight"]
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics

        # Check accuracy (5 correct out of 6 = 0.833)
        assert 0.8 < metrics["accuracy"] < 0.9, "Accuracy should be ~83%"

    @pytest.mark.parametrize("model_name", ["x3d_s"])
    def test_mixed_precision_training(self, model_name, device):
        """Test mixed precision (FP16) training."""
        if device.type != "cuda":
            pytest.skip("Mixed precision requires CUDA")

        config = load_config(f"configs/{model_name}.yaml")
        model = create_model(config)
        model = model.to(device)

        X = torch.randn(4, 3, 16, 224, 224).to(device)
        y = torch.randint(0, 2, (4,)).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()

        model.train()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(X)
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert not torch.isnan(torch.tensor(loss.item()))


@pytest.mark.slow
def test_full_training_run(test_data_structure, device):
    """Integration test: full training pipeline with real data.

    This test is marked as slow and can be skipped with -m "not slow"
    """
    from training.train import train
    import tempfile

    config = load_config("configs/x3d_s.yaml")

    # Override data paths with test data
    config.data.train_dir = test_data_structure["train_dir"]
    config.data.val_dir = test_data_structure["val_dir"]

    # Use temporary directories for checkpoints and logs
    with tempfile.TemporaryDirectory() as tmpdir:
        config.training.checkpoint_dir = str(Path(tmpdir) / "checkpoints")
        config.training.log_dir = str(Path(tmpdir) / "logs")
        config.training.num_epochs = 1  # One epoch for quick test
        config.training.batch_size = 2

        # Run training
        train(config, run_id="test_run_001")

        # Verify outputs
        checkpoint_dir = Path(config.training.checkpoint_dir) / "x3d_s" / "test_run_001"
        log_dir = Path(config.training.log_dir) / "x3d_s" / "test_run_001"

        assert (checkpoint_dir / "final_model.pth").exists(), "Final model not saved"
        assert (log_dir / "training_metrics.json").exists(), "Metrics not saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
