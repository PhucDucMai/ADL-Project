# Fighting Behavior Detection System

A deep learning project for detecting fighting behavior in surveillance videos, with training and inference support for multiple video models, Streamlit visualization, Docker deployment, and reproducible experiment outputs.

## Highlights

- Multi-model training and inference pipeline with one unified config system.
- Streamlit app with two-phase workflow:
  - Full-video detection pass.
  - Annotated video rendering with highlighted fight segments.
- Robust inference checkpoint loading:
  - Supports different checkpoint formats.
  - Handles key prefix variants.
  - Skips incompatible checkpoints and falls back safely.
- Organized experiment outputs by model and run ID.
- GPU-friendly configs for 6GB-class cards.
- Docker setup for both UI inference and training.

## Supported Models

Current model options in this repository:

- x3d_s
- x3d_xs
- r2plus1d_18
- i3d
- slowfast
- videomae
- rtfm
- vad_clip

Main factory:

- [models/factory.py](models/factory.py)

Model configs:

- [configs/default.yaml](configs/default.yaml)
- [configs/x3d_s.yaml](configs/x3d_s.yaml)
- [configs/i3d.yaml](configs/i3d.yaml)
- [configs/slowfast.yaml](configs/slowfast.yaml)
- [configs/videomae.yaml](configs/videomae.yaml)
- [configs/rtfm.yaml](configs/rtfm.yaml)
- [configs/vad_clip.yaml](configs/vad_clip.yaml)

## Project Structure

```text
advantage-dl/
  configs/                  # Model and runtime configurations
  data/                     # Dataset and sample test videos
  models/                   # Model wrappers + model factory
  training/                 # Training pipeline
  inference/                # CLI and inference pipeline
  ui/                       # Streamlit app
  utils/                    # Config, logging, metrics, visualization
  docker/                   # Dockerfile + compose
  tests/                    # Unit and integration tests
  checkpoints/              # Trained weights (ignored in git by pattern)
  logs/                     # Training and runtime logs (ignored by pattern)
```

## Requirements

- Python 3.10+ recommended
- NVIDIA GPU recommended for training/inference speed
- CUDA-compatible PyTorch environment
- FFmpeg runtime (required by PyAV)

Python dependencies are listed in [requirements.txt](requirements.txt).

## Quick Start (Local)

### 1) Create environment and install dependencies

```bash
cd advantage-dl
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Verify core packages

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import av; print(av.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

### 3) Prepare dataset layout

Expected structure:

```text
data/raw/
  train/
    fight/
    normal/
  val/
    fight/
    normal/
  test/
    fight/
    normal/
```

## Training

Run with default config:

```bash
python -m training.train --config configs/default.yaml
```

Run with a specific model config:

```bash
python -m training.train --config configs/slowfast.yaml
python -m training.train --config configs/i3d.yaml
python -m training.train --config configs/videomae.yaml
```

Training entrypoint:

- [training/train.py](training/train.py)

Outputs are organized by model and run ID, for example:

```text
checkpoints/<model_name>/<run_id>/
logs/<model_name>/<run_id>/
```

Typical files include:

- best_model.pth
- final_model.pth
- checkpoint_epoch_N.pth
- training_metrics.json
- metadata.json
- confusion_matrix_epoch_N.png

## Inference

### Option A: Streamlit UI (recommended for demo)

```bash
streamlit run ui/app.py -- --config configs/default.yaml
```

Then open:

- http://localhost:8501

UI entrypoint:

- [ui/app.py](ui/app.py)

### Option B: CLI inference

```bash
python -m inference.run --config configs/default.yaml --source /path/to/video.mp4
```

With overrides:

```bash
python -m inference.run \
  --config configs/slowfast.yaml \
  --source /path/to/video.mp4 \
  --device cuda \
  --threshold 0.65
```

CLI entrypoint:

- [inference/run.py](inference/run.py)

Detector implementation:

- [inference/detector.py](inference/detector.py)

## Docker

Build and run Streamlit service:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Run training service:

```bash
docker compose -f docker/docker-compose.yml run --rm training
```

Docker files:

- [docker/Dockerfile](docker/Dockerfile)
- [docker/docker-compose.yml](docker/docker-compose.yml)

## Recent Stability Improvements

Latest version includes fixes for common deployment and inference failures:

- Streamlit writable HOME/config path handling in containers.
- UID/GID compatibility for container user 1000.
- Safe FPS handling for PyAV encoding path.
- Robust checkpoint compatibility logic across model wrappers.

## Testing

Run all tests:

```bash
pytest -v
```

Run selected tests:

```bash
pytest tests/test_inference.py -v
pytest tests/test_models.py -v
pytest tests/test_training.py -v
```

Test suite:

- [tests/test_inference.py](tests/test_inference.py)
- [tests/test_models.py](tests/test_models.py)
- [tests/test_training.py](tests/test_training.py)
- [tests/test_checkpoint_organization.py](tests/test_checkpoint_organization.py)

## Configuration Guide

Main config blocks:

- model: architecture, pretrained, input size, clip settings
- training: optimizer, lr scheduler, epochs, batch size, checkpoint/log dirs
- data: dataset paths and augmentation
- inference: model path, threshold, device, temporal smoothing

Start from one of:

- [configs/default.yaml](configs/default.yaml)
- [configs/default_gpu6g.yaml](configs/default_gpu6g.yaml)
- [configs/slowfast_gpu6g.yaml](configs/slowfast_gpu6g.yaml)
- [configs/i3d_gpu6g.yaml](configs/i3d_gpu6g.yaml)
- [configs/videomae_gpu6g.yaml](configs/videomae_gpu6g.yaml)
- [configs/vad_clip_gpu6g.yaml](configs/vad_clip_gpu6g.yaml)

## Notes on Repository Size

To keep git history healthy and push-friendly:

- Large checkpoints are ignored by patterns in [.gitignore](.gitignore).
- Generated logs and plots are ignored by patterns in [.gitignore](.gitignore).
- If you need to version very large artifacts, prefer Git LFS.

## Additional Documentation

- [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)
- [docs/TESTING.md](docs/TESTING.md)
- [docs/GPU_MEMORY_OPTIMIZATION_6GB.md](docs/GPU_MEMORY_OPTIMIZATION_6GB.md)
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- [FIXES_APPLIED.md](FIXES_APPLIED.md)

## License

Use this repository according to your project or institutional requirements.
If you plan to open-source publicly, add an explicit LICENSE file.
