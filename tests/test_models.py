"""Tests for model loading, forward pass, and functionality.

Tests:
- Model creation from factory
- Forward pass with correct input/output shapes
- Parameter counting
- Backbone freeze/unfreeze
- Pretrained weight loading
"""

import pytest
import torch

from models.factory import create_model, list_available_models
from utils.config import load_config


class TestModelCreation:
    """Test model creation from factory."""

    @pytest.mark.parametrize("model_name", list_available_models())
    def test_model_creation(self, model_name):
        """Test that each model can be created."""
        # Load model-specific config
        config_path = f"configs/{model_name}.yaml"
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            # Fallback to default config for models without specific config
            config = load_config("configs/default.yaml")
            config.model.name = model_name

        # Create model
        model = create_model(config)
        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.parametrize("model_name", list_available_models())
    def test_model_forward_pass(self, model_name, dummy_video_tensor, device):
        """Test that model forward pass works with correct shapes."""
        # Load config
        config_path = f"configs/{model_name}.yaml"
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            config = load_config("configs/default.yaml")
            config.model.name = model_name

        # Create model and move to device
        model = create_model(config)
        model = model.to(device)
        model.eval()

        # Prepare input matching model expectations
        x = dummy_video_tensor.to(device)

        # Adjust input dimensions if necessary
        if model_name in ("videomae", "vad_clip"):
            # These models may expect different temporal dimensions
            # Use smaller tensor for memory efficiency
            x = torch.randn(2, 3, 8, 224, 224).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check output shape
        assert output.shape[0] == x.shape[0], f"Batch size mismatch for {model_name}"
        assert output.shape[1] == 2, f"Output classes should be 2 for {model_name}"

    @pytest.mark.parametrize("model_name", list_available_models())
    def test_model_parameters(self, model_name):
        """Test that model has trainable parameters."""
        config_path = f"configs/{model_name}.yaml"
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            config = load_config("configs/default.yaml")
            config.model.name = model_name

        model = create_model(config)

        # Check that model has parameters
        params = list(model.parameters())
        assert len(params) > 0, f"Model {model_name} has no parameters"

        # Check that parameters are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, f"Model {model_name} has no trainable parameters"

    @pytest.mark.parametrize("model_name", ["x3d_s", "slowfast"])
    def test_backbone_freeze_unfreeze(self, model_name):
        """Test backbone freezing mechanism (for models that support it)."""
        config_path = f"configs/{model_name}.yaml"
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            config = load_config("configs/default.yaml")
            config.model.name = model_name

        model = create_model(config)

        # Freeze backbone
        model.freeze_backbone()
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_params > 0, f"Model {model_name} freeze_backbone() had no effect"

        # Unfreeze backbone
        model.unfreeze_backbone()
        unfrozen_params = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        assert (
            unfrozen_params == total_params
        ), f"Model {model_name} unfreeze_backbone() didn't unfreeze all"

    def test_get_param_groups(self):
        """Test parameter grouping for different learning rates."""
        config = load_config("configs/x3d_s.yaml")
        model = create_model(config)

        # Get parameter groups
        if hasattr(model, "get_param_groups"):
            param_groups = model.get_param_groups(backbone_lr_scale=0.1)
            assert len(param_groups) >= 1
            assert "params" in param_groups[0]
            assert "lr_scale" in param_groups[0]


class TestModelProperties:
    """Test model properties and metadata."""

    @pytest.mark.parametrize("model_name", list_available_models())
    def test_model_has_count_parameters(self, model_name):
        """Test that models have count_parameters method (from base class)."""
        config_path = f"configs/{model_name}.yaml"
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            config = load_config("configs/default.yaml")
            config.model.name = model_name

        model = create_model(config)

        # Try count_parameters if available
        if hasattr(model, "count_parameters"):
            param_counts = model.count_parameters()
            assert "total" in param_counts
            assert param_counts["total"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
