#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/inference.toml}"
python main.py inference --config "$CONFIG_PATH"
