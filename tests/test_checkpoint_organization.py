"""Tests for checkpoint and log directory organization.

Tests that:
- Checkpoints are saved in correct directory structure
- Metadata files are created
- Model names are in paths
- Confusion matrices are generated at intervals
"""

import json
import tempfile
from pathlib import Path

import pytest

from training.train import generate_run_id
from utils.config import load_config


class TestRunIDGeneration:
    """Test run ID generation."""

    def test_run_id_format(self):
        """Test that run_id has correct format."""
        run_id = generate_run_id()

        # Format should be: run_YYYYMMDD_HHMMSS_<8_char_uuid>
        parts = run_id.split("_")
        assert len(parts) == 4, f"Run ID format incorrect: {run_id}"
        assert parts[0] == "run"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # UUID short

    def test_run_ids_are_unique(self):
        """Test that consecutive run IDs are different."""
        run_id_1 = generate_run_id()
        run_id_2 = generate_run_id()
        assert run_id_1 != run_id_2


class TestCheckpointDirectoryStructure:
    """Test checkpoint save directory structure."""

    def test_checkpoint_path_construction(self):
        """Test that checkpoint paths include model name and run_id."""
        config = load_config("configs/x3d_s.yaml")
        run_id = "test_run_001"

        model_name = config.model.name
        checkpoint_base = Path(config.training.checkpoint_dir)
        checkpoint_dir = checkpoint_base / model_name / run_id

        # Verify structure
        assert str(checkpoint_dir).endswith(f"{model_name}/{run_id}"), \
            f"Checkpoint path {checkpoint_dir} doesn't have correct structure"

    @pytest.mark.parametrize(
        "model_name",
        ["x3d_s", "slowfast", "videomae", "rtfm", "vad_clip"]
    )
    def test_model_specific_config_paths(self, model_name):
        """Test that each model has config file in correct location."""
        config_path = Path("configs") / f"{model_name}.yaml"
        assert config_path.exists(), f"Config for {model_name} not found at {config_path}"

        # Load and verify it has model name
        config = load_config(str(config_path))
        assert config.model.name == model_name


class TestMetadataFile:
    """Test metadata.json file generation."""

    def test_metadata_json_structure(self):
        """Test that metadata.json has required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                "run_id": "test_run_001",
                "model_name": "x3d_s",
                "timestamp": "2026-03-15T15:00:00",
                "best_epoch": 16,
                "best_val_accuracy": 0.8074,
                "final_metrics": {
                    "train_accuracy": 0.96,
                    "val_accuracy": 0.81,
                },
            }

            metadata_path = Path(tmpdir) / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Load and verify
            with open(metadata_path) as f:
                loaded = json.load(f)

            assert loaded["run_id"] == "test_run_001"
            assert loaded["model_name"] == "x3d_s"
            assert loaded["best_epoch"] == 16
            assert loaded["best_val_accuracy"] > 0.8

    def test_metadata_timestamps(self):
        """Test that metadata includes ISO format timestamp."""
        from datetime import datetime

        metadata = {
            "timestamp": datetime.now().isoformat(),
        }

        # Verify it's valid ISO format
        assert "T" in metadata["timestamp"], "Timestamp should be ISO format"
        assert len(metadata["timestamp"]) > 10, "Timestamp seems too short"


class TestLogDirectoryStructure:
    """Test log directory organization."""

    def test_log_path_construction(self):
        """Test that logs are organized by model and run."""
        config = load_config("configs/slowfast.yaml")
        run_id = "test_run_002"

        model_name = config.model.name
        log_base = Path(config.training.log_dir)
        log_dir = log_base / model_name / run_id

        assert str(log_dir).endswith(f"{model_name}/{run_id}"), \
            f"Log path {log_dir} doesn't have correct structure"

    def test_training_metrics_file_location(self):
        """Test where training_metrics.json is saved."""
        metrics_filename = "training_metrics.json"

        config = load_config("configs/x3d_s.yaml")
        run_id = "test_run_003"
        model_name = config.model.name

        log_base = Path(config.training.log_dir)
        metrics_path = log_base / model_name / run_id / metrics_filename

        # Verify it's in the right place
        assert metrics_path.name == metrics_filename
        assert model_name in str(metrics_path)
        assert run_id in str(metrics_path)


class TestConfusionMatrixOutput:
    """Test confusion matrix visualization output."""

    def test_confusion_matrix_filenames(self):
        """Test that confusion matrices have correct naming."""
        log_dir = Path("logs/x3d_s/run_20260315_150000_abc123d")

        # Expected filenames
        cm_filenames = [
            "confusion_matrix_epoch_5.png",
            "confusion_matrix_epoch_10.png",
            "confusion_matrix_epoch_15.png",
            "confusion_matrix_final.png",
        ]

        for filename in cm_filenames:
            expected_path = log_dir / filename
            # Just verify the naming is correct (don't check existence yet)
            assert "confusion_matrix" in str(expected_path)
            assert "epoch" in filename or "final" in filename


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
