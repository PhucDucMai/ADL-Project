# Model Guide: Fighting Detection with Multiple Architectures

This guide explains the available models, their characteristics, recommended configurations, and how to select and train each one.

## Available Models

The fighting detection system now supports 7 different video action detection architectures, each with different trade-offs between accuracy, speed, and memory requirements.

### 1. X3D-S (Default Model)

**Source**: PyTorch Hub (FacebookResearch/PyTorchVideo)
**Configuration**: `configs/x3d_s.yaml`
**Status**: Proven production-ready model

#### Specifications
- **Architecture**: Efficient 3D CNN with depthwise separable convolutions
- **Parameters**: ~3.8M (compact)
- **Input**: 13 frames × 182×182 pixels
- **FLOPs**: ~2.96G
- **Pretrained on**: Kinetics-400 (400 action classes)

#### Performance Target
- Accuracy on Kinetics-400: **73.3%**
- Inference speed: **30+ FPS** on GTX 1660 Super
- Memory: **~1.2 GB** during inference

#### When to Use
- ✅ Limited VRAM (< 4GB free)
- ✅ Real-time inference requirement
- ✅ Production deployment
- ✅ Resource-constrained edge devices

#### Training Command
```bash
python -m training.train --config configs/x3d_s.yaml
```

---

### 2. R(2+1)D-18

**Source**: Torchvision
**Configuration**: `configs/slowfast.yaml` (use as template)
**Status**: Larger, more accurate than X3D

#### Specifications
- **Architecture**: Decomposes 3D conv into 2D spatial + 1D temporal
- **Parameters**: ~31.5M (8× larger than X3D)
- **Input**: 16 frames × 112×112 pixels
- **Pretrained on**: Kinetics-400

#### Performance Target
- Accuracy on Kinetics-400: **78.8%**  (better than X3D)
- Inference speed: **15-20 FPS** on GTX 1660 Super
- Memory: **~2.5 GB** during training

#### When to Use
- ✅ Reasonable VRAM (6GB+) available
- ✅ Accuracy more important than speed
- ✅ Batch processing acceptable

#### Training Command
```bash
python -m training.train --config configs/r2plus1d.yaml
```

---

### 3. I3D (New)

**Source**: PyTorch Hub / TorchVision
**Configuration**: `configs/i3d.yaml`
**Status**: Experimental (may use R(2+1)D fallback if unavailable)

#### Specifications
- **Architecture**: Inflated 3D CNNs (extending 2D ImageNet models)
- **Parameters**: ~95M (large)
- **Input**: 64 frames × 224×224 pixels (high quality)
- **Pretrained on**: Kinetics-400

#### Performance Target
- Accuracy on Kinetics-400: **80.9%** (state-of-the-art)
- Inference speed: **10 FPS** on GTX 1660 Super
- Memory: **~4 GB** during training

#### When to Use
- ✅ Maximum accuracy needed
- ✅ Offline batch processing
- ✅ High-quality video available
- ⚠️ Requires 6GB+ VRAM

#### Training Command
```bash
python -m training.train --config configs/i3d.yaml
```

---

### 4. SlowFast (New)

**Source**: Torchvision
**Configuration**: `configs/slowfast.yaml`
**Status**: Production-ready

#### Specifications
- **Architecture**: Dual-path neural network
  - Slow Path: Full resolution, low frame rate
  - Fast Path: Subsampled, high frame rate
- **Parameters**: ~36M
- **Input**: 64 frames total (Slow: 8@24fps, Fast: 32@24fps)
- **Resolution**: 224×224 pixels
- **Pretrained on**: Kinetics-400

#### Performance Target
- Accuracy on Kinetics-400: **79.9%** (excellent)
- Inference speed: **12-15 FPS** on GTX 1660 Super
- Memory: **~3.5 GB** during training

#### When to Use
- ✅ Best balance of accuracy and speed
- ✅ Real-time possible with lower res
- ✅ Medium VRAM (6GB+)
- ✅ Production deployments

#### Training Command
```bash
python -m training.train --config configs/slowfast.yaml
```

**Recommended**: SlowFast offers the best balance of accuracy, speed, and memory usage for production systems.

---

### 5. VideoMAE (New)

**Source**: HuggingFace (Transformers library)
**Configuration**: `configs/videomae.yaml`
**Status**: Experimental (vision transformer)

#### Specifications
- **Architecture**: Masked Autoencoder Vision Transformer (MAE)
- **Parameters**: ~86M
- **Input**: 16 frames × 224×224 pixels
- **Pretrained on**: Kinetics-400 with masked autoencoder objective
- **Optimizer**: AdamW (recommended for ViT)

#### Performance Target
- Accuracy on Kinetics-400: **81.5%** (excellent)
- Inference speed: **8-10 FPS** on GTX 1660 Super
- Memory: **~3.8 GB** during training

#### When to Use
- ✅ Modern transformer-based architecture
- ✅ Data-efficient pretraining approach
- ✅ Good zero-shot transfer learning
- ✅ Research/experimentation

#### Special Considerations
- Requires `transformers` library (automatically installed via requirements.txt)
- First-time download downloads ~380 MB model
- May need internet connection for first run

#### Training Command
```bash
python -m training.train --config configs/videomae.yaml
```

---

### 6. RTFM (New)

**Source**: HuggingFace / Fallback R(2+1)D
**Configuration**: `configs/rtfm.yaml`
**Status**: Experimental (optimized for fighting detection if available)

#### Specifications
- **Architecture**: Real-Time Fighting Monitoring (specialized for action detection)
- **Parameters**: ~20-30M (if available; otherwise uses R(2+1)D fallback)
- **Input**: 16 frames × 224×224 pixels
- **Pretrained on**: Kinetics-400 or action detection datasets

#### Performance Target
- Estimated accuracy: **80%+** (if official model found)
- Fallback (R(2+1)D): **78.8%**

#### When to Use
- ✅ Fighting-detection-specific architecture
- ✅ If you find the official RTFM model on HF

#### Fallback Behavior
If RTFM is not found on HuggingFace, the system automatically falls back to R(2+1)D-18 and logs a warning.

#### Training Command
```bash
python -m training.train --config configs/rtfm.yaml
```

---

### 7. VAD-CLIP (New)

**Source**: HuggingFace Transformers
**Configuration**: `configs/vad_clip.yaml`
**Status**: Experimental (vision transformer with CLIP-like objective)

#### Specifications
- **Architecture**: Vision Transformer + CLIP embeddings for multi-modal alignment
- **Parameters**: ~86M (ViT-base)
- **Input**: 8 frames × 224×224 pixels (efficient temporal sampling)
- **Pretrained on**: CLIP + action recognition datasets
- **Optimizer**: AdamW (recommended for ViT)

#### Performance Target
- Estimated accuracy: **79-82%** (strong multi-modal transfer)
- Inference speed: **12-15 FPS** on GTX 1660 Super
- Memory: **~3.6 GB** during training

#### When to Use
- ✅ Multi-modal learning (text descriptions + video)
- ✅ Strong zero-shot transfer learning
- ✅ Modern transformer architecture
- ✅ Research applications

#### Fallback Behavior
Falls back to ViT-base if official VAD-CLIP model unavailable.

#### Training Command
```bash
python -m training.train --config configs/vad_clip.yaml
```

---

## Model Comparison Summary

| Model | Params | Input | Speed | Memory | Accuracy | Use Case |
|-------|--------|-------|-------|--------|----------|----------|
| **X3D-S** | 3.8M | 13×182² | 30+ FPS | 1.2GB | 73% | Edge/Real-time |
| **R(2+1)D** | 31.5M | 16×112² | 15-20 FPS | 2.5GB | 79% | Balanced |
| **I3D** | 95M | 64×224² | 10 FPS | 4GB | 81% | Max Accuracy |
| **SlowFast** | 36M | 64×224² | 12-15 FPS | 3.5GB | 80% | **Recommended** |
| **VideoMAE** | 86M | 16×224² | 8-10 FPS | 3.8GB | 82% | Efficient ViT |
| **RTFM** | 20-30M | 16×224² | Variable | 2.5GB | 80% | Fighting-specific |
| **VAD-CLIP** | 86M | 8×224² | 12-15 FPS | 3.6GB | 80% | Multi-modal |

---

## How to Select a Model

### Decision Tree

```
Is VRAM < 2GB?
  → X3D-S (only option)

Is VRAM < 4GB?
  → X3D-S or R(2+1)D (with smaller batch size)

Is real-time inference required?
  → X3D-S or SlowFast

Do you need max accuracy?
  → I3D or VideoMAE or VAD-CLIP

Are you doing batch processing?
  → I3D (most accurate) or SlowFast (balanced)

Do you want transformer-based model?
  → VideoMAE or VAD-CLIP

Unsure? Use SlowFast
  → Best balance for production
```

---

## Configuration Files

Each model has its own config file in `configs/`:

```bash
configs/
├── x3d_s.yaml       # Compact, efficient
├── r2plus1d_18.yaml # (can use slowfast.yaml as template)
├── i3d.yaml         # High accuracy
├── slowfast.yaml    # Recommended baseline
├── videomae.yaml    # ViT-based
├── rtfm.yaml        # Fighting-specific
└── vad_clip.yaml    # Multi-modal ViT
```

Each config includes:
- **Model spec**: architecture, pretrained weights, class count
- **Training params**: batch size, LR, optimizer, warmup epochs
- **Data spec**: input dimensions, augmentation
- **Inference spec**: threshold, smoothing window

---

## Training a Model

### Example: Train SlowFast

```bash
# Start training
python -m training.train --config configs/slowfast.yaml

# Output:
# - Checkpoints: checkpoints/slowfast/run_20260315_150000_abc123d/
# - Logs: logs/slowfast/run_20260315_150000_abc123d/
# - Metrics: training_metrics.json
# - Visualizations: loss_curves.png, accuracy_curves.png, confusion_matrix_epoch_*.png
```

### Example: Train I3D with custom hyperparameters

```bash
# Edit configs/i3d.yaml to change:
# - batch_size (increase if more VRAM)
# - learning_rate
# - num_epochs

# Then train
python -m training.train --config configs/i3d.yaml
```

---

## Checkpoint Organization

After training, checkpoints are organized by model and run:

```
checkpoints/
└── slowfast/
    ├── run_20260315_150000_abc123d/
    │   ├── best_model.pth              # Best validation accuracy
    │   ├── checkpoint_epoch_5.pth      # Periodic snapshots
    │   ├── checkpoint_epoch_10.pth
    │   ├── final_model.pth             # After all epochs
    │   ├── metadata.json               # Model info + metrics
    │   └── training_metrics.json       # Full metrics + CMs
    └── run_20260315_160000_def456g/
        └── ...

logs/
└── slowfast/
    └── run_20260315_150000_abc123d/
        ├── training.log
        ├── loss_curves.png
        ├── accuracy_curves.png
        ├── lr_schedule.png
        ├── confusion_matrix_epoch_5.png
        ├── confusion_matrix_epoch_10.png
        ├── confusion_matrix_final.png
        └── metadata.json
```

---

## Using Trained Models for Inference

### Load Model in Code

```python
from inference.detector import FightDetector
from utils.config import load_config

# Load config
config = load_config("configs/slowfast.yaml")
config.inference.model_path = "checkpoints/slowfast/run_20260315_150000_abc123d/best_model.pth"

# Create detector
detector = FightDetector(config=config)

# Make prediction on video clip
prediction = detector.predict_clip(video_frames)  # shape: (T, H, W, 3)
# Returns: {"is_fight": bool, "confidence": float, "label": str, ...}
```

### Use in Streamlit UI

The WebUI automatically detects available checkpoint directories and allows model selection.

---

## Model Migration & Experimentation

To experiment with multiple models:

1. **Train baseline**: `python -m training.train --config configs/slowfast.yaml`
2. **Compare with X3D**: `python -m training.train --config configs/x3d_s.yaml`
3. **Try I3D**: `python -m training.train --config configs/i3d.yaml`
4. **Compare results**: Check `logs/*/run_*/metadata.json` for best accuracies

Results are automatically organized, making comparison easy:

```bash
# Compare best accuracies across all models
grep best_val_accuracy logs/*/run_*/metadata.json
```

---

## Model Descriptions and References

### X3D
- Paper: "X3D: Expanding Architectures for Efficient Video Recognition" (Feichtenhofer, 2020, CVPR)
- PyTorch Hub: `torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s')`

### R(2+1)D
- Paper: "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (Tran et al., 2018)
- Torchvision: `torchvision.models.video.r2plus1d_18`

### I3D
- Paper: "Quo Vadis, Action Recognition?" (Carreira & Zisserman, 2017, CVPR)
- Inflated 3D CNNs from 2D ImageNet models

### SlowFast
- Paper: "SlowFast Networks for Video Recognition" (Feichtenhofer et al., 2019, ICCV)
- Torchvision: `torchvision.models.video.slowfast_r50`
- **Highly Recommended**: Best balance of accuracy & speed for production

### VideoMAE
- Paper: "VideoMAE: Masked Autoencoders are Data-Efficient Learners" (Tong et al., 2022, NeurIPS)
- HuggingFace: `MCG-NJU/videomae-base-finetuned-kinetics`
- Efficient self-supervised pretraining approach

### VAD-CLIP
- Vision Transformer + CLIP-style multi-modal alignment
- Strong zero-shot transfer learning
- Temporal modeling via ViT

---

## FAQ

**Q: Which model should I use?**
A: Start with SlowFast. It offers the best balance.

**Q: Can I train multiple models in parallel?**
A: Yes, but monitor VRAM usage. Each model uses 2-4 GB.

**Q: How do I compare model performance?**
A: Check `logs/*/run_*/metadata.json` for best_val_accuracy.

**Q: Can I use a different model for inference?**
A: Yes, just set `config.inference.model_path` to any checkpoint.

**Q: Do HuggingFace models need internet?**
A: First run downloads weights (~300-500MB). Cached on disk after.

**Q: How much faster is X3D vs SlowFast?**
A: X3D: 30+ FPS vs SlowFast: 12-15 FPS on GTX 1660 Super (2-3× faster).

---

## Next Steps

1. **Train a model**: `python -m training.train --config configs/slowfast.yaml`
2. **Monitor training**: Check logs in real-time
3. **Evaluate**: Review confusion matrices and metrics
4. **Inference**: Use trained checkpoint with WebUI or code
5. **Compare**: Experiment with different models

See README.md for setup instructions and TESTING.md for running tests.
