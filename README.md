# Fighting Behavior Detection System

Real-time detection of fighting behavior in video surveillance using deep learning.

## Table of Contents

- [Problem Description](#problem-description)
- [System Architecture](#system-architecture)
- [Research Summary](#research-summary)
- [Chosen Approach](#chosen-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Guide](#training-guide)
- [Inference Guide](#inference-guide)
- [Docker Deployment](#docker-deployment)
- [Configuration Reference](#configuration-reference)

---

## Problem Description

Abnormal crowd behavior detection is a critical task for public safety and surveillance. Fighting is one of the most common and dangerous forms of abnormal behavior in public spaces such as streets, transit stations, schools, and shopping centers.

Manual monitoring of surveillance feeds is impractical at scale. This system automates the detection of fighting behavior using a deep learning model that processes video frames in real-time, alerting operators when violence is detected.

**Application scenarios:**
- Public transportation surveillance (bus, metro stations)
- School and campus security monitoring
- Shopping mall and retail security
- Street and parking lot surveillance
- Prison and detention facility monitoring

---

## System Architecture

```
+------------------+     +------------------+     +------------------+
|   Video Source   | --> |  Frame Reader    | --> |  Frame Buffer    |
|  (RTSP / File)   |     |  (PyAV/FFmpeg)   |     |  (Thread-safe)   |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
+------------------+     +------------------+     +------------------+
|   Display /      | <-- | Post-Processing  | <-- |  Model Inference |
|   Warning UI     |     | (Smoothing)      |     |  (X3D-S / GPU)   |
+------------------+     +------------------+     +------------------+
```

**Pipeline stages:**

1. **Frame Acquisition**: PyAV (FFmpeg bindings) reads frames from RTSP streams or video files in a background thread. FFmpeg is used instead of OpenCV for better performance, codec support, and lower latency with RTSP streams.

2. **Frame Buffering**: A thread-safe circular buffer stores recent frames. The inference module samples clips from this buffer at configurable intervals.

3. **Clip Extraction**: A temporal clip of N frames (with stride) is sampled from the buffer, resized, and normalized for model input.

4. **Model Inference**: The X3D-S model processes the clip and outputs class probabilities (normal vs. fight). Mixed precision (FP16) is used for speed and memory efficiency.

5. **Temporal Smoothing**: Predictions are averaged over a sliding window to reduce false positives from momentary misclassifications.

6. **Display**: Results are overlaid on the video frame. A red warning banner is displayed when fighting is detected with sufficient confidence.

---

## Research Summary

### A. Video-based Action Recognition

| Method | How It Works | Strengths | Weaknesses | Params | Real-time |
|--------|-------------|-----------|------------|--------|-----------|
| **SlowFast** | Two-pathway network: slow pathway captures spatial features at low frame rate; fast pathway captures motion at high frame rate | High accuracy; captures both spatial and temporal cues well | High memory usage (~34M params); requires two pathways | ~34M | Moderate |
| **Video Swin Transformer** | 3D shifted window attention mechanism adapted from image Swin Transformer | State-of-the-art accuracy on Kinetics; strong temporal modeling | Very heavy (>80M params); requires large GPU memory; slow inference | >80M | No |
| **I3D** | Inflates 2D ImageNet-pretrained convolutions to 3D for spatiotemporal features | Good accuracy; conceptually simple; strong transfer learning | Large model (~25M); relatively slow; older architecture | ~25M | Moderate |
| **X3D** | Progressively expands a tiny base architecture along multiple axes (temporal, spatial, width, depth) | Very efficient (~3.8M params); good accuracy/speed tradeoff; designed for mobile/edge | Slightly lower peak accuracy than larger models | ~3.8M | Yes |

### B. Skeleton-based Action Recognition

| Method | How It Works | Strengths | Weaknesses | Params | Real-time |
|--------|-------------|-----------|------------|--------|-----------|
| **ST-GCN** | Graph Convolutional Network on skeleton sequences; spatial graph represents body joints, temporal edges connect joints across frames | Lightweight; robust to appearance changes; interpretable | Requires separate pose estimator; sensitive to pose estimation errors | ~3M | Yes (excluding pose) |
| **CTR-GCN** | Refines channel-wise topology of the graph dynamically; learns different graph structures per channel | Better than ST-GCN; dynamic topology adapts to actions | More complex than ST-GCN; still needs pose estimator | ~3.5M | Yes (excluding pose) |
| **PoseC3D** | Converts pose keypoints to 3D heatmap volumes and applies 3D CNN | Combines strengths of skeleton and appearance methods; robust | Heavier than pure GCN approaches; needs pose estimation | ~10M | Moderate |

### C. Skeleton-based Spatio-Temporal Action Detection

These methods (e.g., combining person detectors + pose estimators + action classifiers) can localize actions per-person in a scene. They are the most flexible for multi-person scenarios but require a multi-stage pipeline (detection -> tracking -> pose -> classification), making them complex to deploy and harder to run in real-time on modest hardware.

### D. Video Anomaly Detection

Unsupervised/semi-supervised approaches that learn "normal" patterns and flag deviations:

| Method | How It Works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Autoencoder-based** | Learns to reconstruct normal video; high reconstruction error indicates anomaly | No labeled anomaly data needed; generalizes to unseen anomalies | High false positive rate; cannot distinguish anomaly types |
| **Prediction-based** | Predicts future frames; prediction error serves as anomaly score | Self-supervised; captures temporal dynamics | Sensitive to video quality; struggles with complex scenes |
| **Memory-augmented** | Stores prototype normal patterns in memory; queries against incoming frames | Better than vanilla autoencoders; more robust | Still cannot classify specific anomaly types; threshold tuning required |

---

## Chosen Approach

### Recommendation: X3D-S (Expanded 3D Networks - Small)

**Why X3D-S was chosen over alternatives:**

1. **Memory efficiency**: At ~3.8M parameters, X3D-S fits comfortably in 6GB VRAM with batch size 8 during training and leaves headroom for video decoding during inference.

2. **Real-time capability**: X3D was explicitly designed for efficient video understanding. On a GTX 1660 Super, it can process clips at 30+ FPS with FP16 inference.

3. **Strong pretrained features**: Pretrained on Kinetics-400 (400 action classes), the backbone already understands human actions, motion patterns, and temporal dynamics. Fine-tuning for binary fight/normal classification converges quickly.

4. **Input efficiency**: X3D-S uses only 13 frames at 182x182 resolution, requiring minimal video buffering and preprocessing compared to models needing 32 or 64 frames.

5. **Simpler pipeline than skeleton methods**: Skeleton-based methods require a separate pose estimator (e.g., HRNet, ViTPose) that adds latency and a failure point. X3D-S operates directly on RGB frames.

6. **Supervised classification over anomaly detection**: For the specific task of fighting detection (binary classification), supervised learning with labeled data provides higher precision and recall than unsupervised anomaly detection methods that cannot distinguish fighting from other unusual events.

**Fallback option**: R(2+1)D-18 from torchvision is provided as an alternative. It is simpler to set up (no pytorchvideo dependency) but is larger (~31M params) and less efficient.

---

## Project Structure

```
advantage-dl/
|
|-- configs/
|   |-- default.yaml          # Default configuration
|
|-- data/
|   |-- raw/                   # Raw video data
|   |   |-- train/
|   |   |   |-- fight/         # Fighting video clips
|   |   |   |-- normal/        # Normal video clips
|   |   |-- val/
|   |       |-- fight/
|   |       |-- normal/
|   |-- processed/             # Preprocessed data (optional)
|   |-- dataset.py             # PyTorch Dataset class
|   |-- transforms.py          # Video augmentation transforms
|   |-- video_reader.py        # FFmpeg-based video reader (PyAV)
|
|-- models/
|   |-- x3d.py                 # X3D model wrapper
|   |-- r2plus1d.py            # R(2+1)D fallback model
|   |-- factory.py             # Model creation factory
|
|-- training/
|   |-- train.py               # Training script
|
|-- inference/
|   |-- detector.py            # Fight detection wrapper
|   |-- stream_reader.py       # Threaded RTSP/file reader
|   |-- pipeline.py            # Complete inference pipeline
|   |-- run.py                 # CLI entry point for inference
|
|-- ui/
|   |-- app.py                 # Streamlit web interface
|
|-- utils/
|   |-- config.py              # Configuration management
|   |-- logger.py              # Logging setup
|   |-- metrics.py             # Training metrics tracking
|   |-- visualization.py       # Plot training curves
|
|-- docker/
|   |-- Dockerfile             # Docker image definition
|   |-- docker-compose.yml     # Compose configuration
|
|-- checkpoints/               # Saved model weights
|-- logs/                      # Training logs and plots
|-- requirements.txt           # Python dependencies
|-- README.md                  # This file
```

---

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (tested on GTX 1660 Super)
- NVIDIA drivers and CUDA toolkit (11.8 recommended)
- FFmpeg (system library)

### Local Installation

```bash
# Clone or navigate to the project
cd advantage-dl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install FFmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import av; print('PyAV version:', av.__version__)"
python -c "import pytorchvideo; print('PyTorchVideo available')"
```

---

## Dataset Preparation

### Recommended Dataset: RWF-2000

The [RWF-2000 dataset](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) contains 2000 video clips (1000 fight, 1000 normal) captured from surveillance cameras.

### Directory Structure

Organize your data as follows:

```
data/raw/
|-- train/
|   |-- fight/
|   |   |-- video001.avi
|   |   |-- video002.avi
|   |   |-- ...
|   |-- normal/
|       |-- video001.avi
|       |-- video002.avi
|       |-- ...
|-- val/
    |-- fight/
    |   |-- ...
    |-- normal/
        |-- ...
```

### Using RWF-2000

1. Download the RWF-2000 dataset
2. Extract the archive
3. The dataset already comes split into train/val with fight/normal subdirectories
4. Copy or symlink the directories into `data/raw/`

```bash
# Example with symlinks
ln -s /path/to/RWF-2000/train data/raw/train
ln -s /path/to/RWF-2000/val data/raw/val
```

### Using Custom Data

- Video clips should be 2-10 seconds long
- Supported formats: .avi, .mp4, .mkv, .mov, .wmv
- Minimum resolution: 224x224 recommended
- Split ratio: 80% train, 20% validation recommended

---

## Training Guide

### Basic Training

```bash
python -m training.train --config configs/default.yaml
```

### Custom Configuration

Edit `configs/default.yaml` or create a new config file:

```yaml
model:
  name: "x3d_s"       # Model architecture
  num_classes: 2       # fight / normal
  pretrained: true     # Use Kinetics-400 pretrained weights

training:
  batch_size: 8        # Reduce to 4 if OOM on 6GB VRAM
  num_epochs: 30
  learning_rate: 0.005
  mixed_precision: true  # FP16 training for memory savings

data:
  train_dir: "data/raw/train"
  val_dir: "data/raw/val"
  clip_length: 13      # Frames per clip
  frame_stride: 2      # Temporal stride
  spatial_size: 182    # Input resolution
```

### Training Strategy

The training pipeline uses a two-phase approach:

1. **Phase 1 (Epochs 1-5)**: Backbone is frozen, only the classification head is trained. This allows the head to adapt to the new task without disrupting pretrained features.

2. **Phase 2 (Epochs 6-30)**: All layers are unfrozen for full fine-tuning with a lower learning rate. Cosine annealing schedule gradually reduces the learning rate.

### Training Outputs

After training, the following files are produced:

```
checkpoints/
|-- best_model.pth          # Best validation loss checkpoint
|-- final_model.pth         # Final epoch checkpoint
|-- checkpoint_epoch_*.pth  # Periodic checkpoints

logs/
|-- loss_curves.png         # Training/validation loss plot
|-- accuracy_curves.png     # Training/validation accuracy plot
|-- lr_schedule.png         # Learning rate schedule plot
|-- confusion_matrix.png    # Final validation confusion matrix
|-- training_metrics.json   # All metrics in JSON format
```

### Memory Optimization

If you encounter out-of-memory errors:

1. Reduce `batch_size` to 4 or 2
2. Ensure `mixed_precision: true` is set
3. Reduce `num_workers` to 2
4. Consider using `x3d_xs` (even smaller model)

---

## Inference Guide

### Command-Line Inference

```bash
# From a video file
python -m inference.run --source /path/to/video.mp4 --config configs/default.yaml

# From an RTSP stream
python -m inference.run --source rtsp://192.168.1.100:554/stream --config configs/default.yaml

# With specific model checkpoint
python -m inference.run --source video.mp4 --model checkpoints/best_model.pth

# Adjust confidence threshold
python -m inference.run --source video.mp4 --threshold 0.7
```

### Streamlit Web Interface

```bash
streamlit run ui/app.py -- --config configs/default.yaml
```

Then open the displayed URL (default: http://localhost:8501) in a browser.

The web interface allows you to:
- Select input source (video file path, file upload, or RTSP URL)
- Adjust the confidence threshold
- Start and stop detection
- View real-time video with detection overlays

### RTSP Stream Setup

For IP cameras, the RTSP URL typically follows this format:

```
rtsp://<username>:<password>@<ip>:<port>/<path>
```

Examples:
```
rtsp://admin:admin123@192.168.1.100:554/stream1
rtsp://192.168.1.100:554/live/ch0
```

Test RTSP connectivity with FFmpeg before running detection:
```bash
ffplay rtsp://192.168.1.100:554/stream
```

---

## Docker Deployment

### Prerequisites

- Docker Engine 20.10+
- NVIDIA Container Toolkit (for GPU support)

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build and Run

```bash
cd docker

# Build the image
docker compose build

# Run the web interface
docker compose up fight-detection

# Run training
docker compose --profile training run training
```

### Access the UI

Open http://localhost:8501 in your browser after starting the container.

### For RTSP Streams in Docker

If accessing RTSP streams from within the container, you may need to use host networking. Uncomment the `network_mode: host` line in `docker-compose.yml`.

---

## Configuration Reference

All settings are managed through YAML configuration files in `configs/`.

### Model Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `x3d_s` | Model architecture (`x3d_s`, `x3d_xs`, `r2plus1d_18`) |
| `model.num_classes` | `2` | Number of output classes |
| `model.pretrained` | `true` | Load Kinetics-400 pretrained weights |
| `model.clip_length` | `13` | Number of input frames per clip |
| `model.spatial_size` | `182` | Input spatial resolution |
| `model.dropout_rate` | `0.5` | Dropout before classification head |

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.batch_size` | `8` | Training batch size |
| `training.num_epochs` | `30` | Total training epochs |
| `training.learning_rate` | `0.005` | Initial learning rate |
| `training.mixed_precision` | `true` | Enable FP16 training |
| `training.freeze_backbone_epochs` | `5` | Epochs to freeze backbone |
| `training.label_smoothing` | `0.1` | Label smoothing factor |
| `training.gradient_clip_norm` | `1.0` | Max gradient norm (0 to disable) |

### Inference Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference.confidence_threshold` | `0.6` | Min confidence for fight detection |
| `inference.buffer_size` | `64` | Frame buffer size |
| `inference.inference_interval` | `8` | Run inference every N frames |
| `inference.temporal_smoothing_window` | `3` | Average over N predictions |
| `inference.warning_display_frames` | `30` | Duration of warning display |

---

## Hardware Requirements

**Minimum (tested):**
- CPU: Intel Core i3-12400F
- GPU: NVIDIA GTX 1660 Super (6GB VRAM)
- RAM: 16GB

**Recommended:**
- CPU: Intel Core i5/i7 or AMD Ryzen 5/7
- GPU: NVIDIA RTX 3060+ (8GB+ VRAM)
- RAM: 32GB

**Memory usage estimates (X3D-S):**
- Training (batch_size=8, FP16): ~4GB VRAM
- Inference (single clip, FP16): ~1.5GB VRAM
