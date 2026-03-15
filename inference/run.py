"""Command-line entry point for running inference."""

import argparse
import logging

from utils.config import load_config
from utils.logger import setup_logger
from inference.pipeline import run_inference


def main():
    parser = argparse.ArgumentParser(description="Run fighting detection inference")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Video source: RTSP URL (rtsp://...) or path to video file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint (overrides config)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use for inference (e.g., cuda, cpu)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold for fight detection",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        config.inference.device = args.device
    if args.threshold:
        config.inference.confidence_threshold = args.threshold

    setup_logger("root", log_dir="logs")
    setup_logger(__name__, log_dir="logs")

    run_inference(config, args.source, args.model)


if __name__ == "__main__":
    main()
