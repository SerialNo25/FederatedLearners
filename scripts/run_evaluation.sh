#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/evaluation.toml}"
python main.py evaluation --config "$CONFIG_PATH"
