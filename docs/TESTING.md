# Testing Guide

This guide explains how to run, write, and understand the test suite for the fighting detection system.

## Quick Start

### Install Test Dependencies

Test dependencies are already in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install pytest explicitly:

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_models.py -v
```

### Run Tests for Specific Model

```bash
pytest tests/test_models.py -k "x3d_s" -v
```

---

## Test Organization

```
tests/
├── conftest.py                         # Pytest configuration and fixtures
├── test_models.py                      # Model loading and inference tests
├── test_training.py                    # Training pipeline tests (marked as @slow)
├── test_inference.py                   # Inference pipeline tests
├── test_checkpoint_organization.py     # Directory structure and metadata tests
└── __init__.py
```

---

## Running Different Test Categories

### Fast Tests (Model Loading & Inference)

```bash
# Run only fast tests (excludes @slow marked tests)
pytest tests/ -m "not slow"

# Takes ~1-2 minutes for all models
```

### Slow Tests (Full Training)

```bash
# Run slow integration tests
pytest tests/ -m slow

# Takes 10-20 minutes depending on VRAM
# Only runs 1-epoch training for speed
```

### GPU-Only Tests

```bash
# Run only tests that require GPU
pytest tests/ -m gpu

# Skips CPU-only tests if no GPU available
```

### All Tests

```bash
# Run everything
pytest tests/ -v

# Total time: 15-30 minutes
```

---

## Test Files Reference

### conftest.py - Pytest Fixtures

Provides reusable fixtures for all tests:

| Fixture | Purpose | Scope |
|---------|---------|-------|
| `device` | Gets CUDA or CPU | Session |
| `tmp_data_dir` | Temporary directory | Session |
| `dummy_video_tensor` | Random video (2,3,16,224,224) | Function |
| `dummy_labels` | Random labels [0,1] | Function |
| `sample_config` | Loaded default config | Function |
| `test_video_path` | Creates test video file | Function |
| `test_data_structure` | Creates train/val data dirs | Function |
| `model_configs` | List of all model names | Function |

Example usage:

```python
def test_something(dummy_video_tensor, device):
    # dummy_video_tensor is automatically passed
    assert dummy_video_tensor.shape == (2, 3, 16, 224, 224)
    assert device.type in ("cuda", "cpu")
```

---

### test_models.py - Model Tests

Tests model creation, forward pass, and properties.

#### Tests Included

```
TestModelCreation
  ├─ test_model_creation (parametrized for all models)
  ├─ test_model_forward_pass
  ├─ test_model_parameters
  ├─ test_backbone_freeze_unfreeze
  └─ test_get_param_groups

TestModelProperties
  └─ test_model_has_count_parameters
```

#### Run Model Tests

```bash
# All model tests
pytest tests/test_models.py -v

# Only X3D tests
pytest tests/test_models.py::TestModelCreation::test_model_creation[x3d_s] -v

# Only forward pass tests
pytest tests/test_models.py -k "forward_pass" -v
```

#### What Gets Tested

1. ✅ Each model can be created from factory
2. ✅ Forward pass produces correct output shape (B, 2)
3. ✅ Models have trainable parameters
4. ✅ Backbone freeze/unfreeze works
5. ✅ Parameter grouping for LR scaling

---

### test_training.py - Training Tests

Tests training loop, checkpoints, and metrics computation.

#### Tests Included

```
TestTrainingPipeline (marked @slow)
  ├─ test_single_epoch_training (parametrized)
  ├─ test_checkpoint_saving_and_loading
  ├─ test_metrics_computation
  ├─ test_mixed_precision_training
  └─ test_full_training_run

Standalone Tests
  └─ test_full_training_run (integration test)
```

#### Run Training Tests

```bash
# Only fast training tests (excluding @slow)
pytest tests/test_training.py -m "not slow"

# Only slow tests
pytest tests/test_training.py -m slow

# Specific slow test
pytest tests/test_training.py::TestTrainingPipeline::test_single_epoch_training[x3d_s] -m slow
```

#### What Gets Tested

1. ✅ Training loop runs for 1 epoch without errors
2. ✅ Loss decreases (basic sanity check)
3. ✅ Checkpoints are saved correctly
4. ✅ Metrics are computed (accuracy, precision, recall, F1)
5. ✅ Mixed precision (FP16) training works on CUDA
6. ✅ Full pipeline with real data (train + val)

#### Sample Output

```
tests/test_training.py::TestTrainingPipeline::test_single_epoch_training[x3d_s] PASSED [42%]
tests/test_training.py::TestTrainingPipeline::test_checkpoint_saving_and_loading PASSED [50%]
tests/test_training.py::test_full_training_run PASSED [100%]
```

---

### test_inference.py - Inference Tests

Tests model inference, predictor classes, and confidence thresholding.

#### Tests Included

```
TestInferencePipeline
  ├─ test_model_checkpoint_saving_and_loading
  ├─ test_fight_detector_initialization
  ├─ test_predict_clip_shape
  └─ test_confidence_thresholding

TestInferenceOnRealVideo (marked @slow)
  └─ test_inference_on_test_video

TestTemporalSmoothing
  └─ test_deque_based_smoothing
```

#### Run Inference Tests

```bash
# Fast inference tests
pytest tests/test_inference.py -m "not slow" -v

# With real video (slow)
pytest tests/test_inference.py::TestInferenceOnRealVideo -m slow

# Just confidence tests
pytest tests/test_inference.py -k "confidence" -v
```

#### What Gets Tested

1. ✅ Checkpoint can be saved and loaded
2. ✅ FightDetector initializes without errors
3. ✅ Predict produces correct output shape
4. ✅ Confidence values are in [0, 1]
5. ✅ Temporal smoothing averages predictions correctly
6. ✅ Inference works on actual video files

---

### test_checkpoint_organization.py - Directory Tests

Tests run_id generation and checkpoint/log directory organization.

#### Tests Included

```
TestRunIDGeneration
  ├─ test_run_id_format
  └─ test_run_ids_are_unique

TestCheckpointDirectoryStructure
  ├─ test_checkpoint_path_construction
  └─ test_model_specific_config_paths

TestMetadataFile
  ├─ test_metadata_json_structure
  └─ test_metadata_timestamps

TestLogDirectoryStructure
  ├─ test_log_path_construction
  ├─ test_training_metrics_file_location
  └─ TestConfusionMatrixOutput

TestConfusionMatrixOutput
  └─ test_confusion_matrix_filenames
```

#### Run Checkpoint Tests

```bash
pytest tests/test_checkpoint_organization.py -v

# Run ID generation tests only
pytest tests/test_checkpoint_organization.py::TestRunIDGeneration -v
```

#### What Gets Tested

1. ✅ Run IDs have correct format (run_YYYYMMDD_HHMMSS_uuid)
2. ✅ Run IDs are unique
3. ✅ Checkpoints saved to `checkpoints/<model>/<run_id>/`
4. ✅ Logs saved to `logs/<model>/<run_id>/`
5. ✅ metadata.json contains required fields
6. ✅ Confusion matrices named correctly

---

## Parametrized Tests

Many tests run multiple times with different parameters:

```python
@pytest.mark.parametrize("model_name", list_available_models())
def test_model_creation(self, model_name):
    # This test runs once for each model
```

To run only specific parameter values:

```bash
# Test only X3D-S and SlowFast
pytest tests/test_models.py::TestModelCreation::test_model_creation -k "x3d_s or slowfast" -v

# Test only models from torchvision
pytest tests/test_models.py -k "slowfast or r2plus1d" -v
```

---

## Fixtures in Detail

### Device Selection

```python
def test_model_on_device(device):
    # device is 'cuda' if available, else 'cpu'
    model = create_model(config)
    model = model.to(device)
    assert str(device) in ("cpu",  "cuda:0")
```

### Temporary Directories

```python
def test_save_checkpoint(tmp_data_dir):
    # tmp_data_dir is a Path to temporary directory
    checkpoint_path = tmp_data_dir / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    assert checkpoint_path.exists()
```

### Test Video Creation

```python
def test_inference_on_video(test_video_path):
    # test_video_path returns Path to 10-frame test video
    container = av.open(str(test_video_path))
    assert len(list(container.decode(video=0))) == 10
```

### Test Data Structure

```python
def test_training_with_data(test_data_structure):
    # Creates: train/{fight,normal}/, val/{fight,normal}/
    assert Path(test_data_structure["train_dir"]).exists()
    assert Path(test_data_structure["val_dir"]).exists()
```

---

## Writing Your Own Tests

### Test Template

```python
import pytest
from models.factory import create_model
from utils.config import load_config

class TestMyFeature:
    """Test my new feature."""

    def test_something_basic(self):
        """Test basic functionality."""
        config = load_config("configs/x3d_s.yaml")
        model = create_model(config)
        assert model is not None

    @pytest.mark.parametrize("model_name", ["x3d_s", "slowfast"])
    def test_with_multiple_models(self, model_name):
        """Test across multiple models."""
        # Test runs twice: once with x3d_s, once with slowfast
        pass

    @pytest.mark.slow
    def test_slow_operation(self):
        """This test is slow and can be skipped."""
        # Only runs if explicitly requested with -m slow
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_gpu_feature(self):
        """Skip if GPU not available."""
        pass
```

### Running Your Test

```bash
# Run just your new test
pytest tests/test_myfile.py::TestMyFeature::test_something_basic -v

# With verbose output
pytest tests/test_myfile.py -vv
```

---

## Coverage Analysis

Generate test coverage report:

```bash
pytest tests/ --cov=models --cov=training --cov=inference --cov=utils --cov-report=html

# Opens htmlcov/index.html with coverage breakdown
```

---

## Debugging Failed Tests

### Verbose Output

```bash
pytest tests/test_models.py::TestModelCreation::test_model_forward_pass[x3d_s] -vv
```

### Print Statements

```python
def test_something(self):
    print("Debug message")  # Visible with -s flag
    pytest.main([__file__, "-v", "-s"])
```

### Stop on First Failure

```bash
pytest tests/ -x  # Stop on first failure
```

### Drop Into Debugger

```bash
pytest tests/ --pdb  # Opens pdb on failure
```

---

## Continuous Integration

To run tests before committing:

```bash
# Run fast tests only
pytest tests/ -m "not slow"

# Or use pre-commit hook (if configured)
git commit  # Runs tests automatically
```

---

## Test Statistics

Current test coverage:

| Category | Count | Avg Time |
|----------|-------|----------|
| Model tests | 25+ | 30 sec |
| Training tests | 8 | 120 sec |
| Inference tests | 12 | 45 sec |
| Checkpoint tests | 15 | 5 sec |
| **Total** | **60+** | **~3 min** (fast mode) |

With slow tests: ~20-30 min total

---

## Common Issues & Solutions

### Issue: "CUDA out of memory"
```bash
# Run on CPU only
pytest tests/ -m "not gpu"

# Or reduce batch size in conftest.py
```

### Issue: "ModuleNotFoundError: No module named 'models'"
```bash
# Run from project root
cd /path/to/project
pytest tests/
```

### Issue: "Test hangs on HuggingFace download"
```bash
# Video MAE/VAD-CLIP download models on first run
# Skip HF tests on first run with no internet
pytest tests/ -k "not videomae and not vad_clip"
```

---

## Next Steps

1. **Run fast tests**: `pytest tests/ -m "not slow"`
2. **Review coverage**: `pytest tests/ --cov=models`
3. **Add your tests**: Create new test_*.py file
4. **Run before commit**: `pytest tests/ -m "not slow"`

See MODEL_GUIDE.md for architecture details and README.md for setup.
