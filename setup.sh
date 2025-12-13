#!/bin/bash

# Setup script for griffin-tele
# Sets up uv environment and downloads required MediaPipe models

set -e  # Exit on error

echo "Setting up griffin-tele..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ uv installed"
    echo ""
fi

# Sync dependencies with uv
echo "Syncing Python dependencies with uv..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Download Pose Landmarker Model
echo "Downloading Pose Landmarker Model..."
curl -o pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
echo "✓ Pose Landmarker Model downloaded"
echo ""

# Download Hand Landmarker Model
echo "Downloading Hand Landmarker Model..."
curl -o hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
echo "✓ Hand Landmarker Model downloaded"
echo ""

echo "Setup complete! You can now run:"
echo "  uv run griffin-teleop --mock --camera 0"
echo "or with real robot:"
echo "  uv run griffin-teleop --can can0 --camera 0"
