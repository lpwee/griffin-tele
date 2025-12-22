#!/bin/bash

# Setup script for griffin-tele
# Sets up uv environment and downloads required MediaPipe models

set -e  # Exit on error

echo "Setting up griffin-tele..."
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=macOS;;
    *)          PLATFORM=Unknown;;
esac
echo "Detected platform: $PLATFORM"
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

# Install RGB-D dependencies on Linux
if [ "$PLATFORM" = "Linux" ]; then
    echo "Installing RGB-D camera support (pyorbbecsdk)..."
    if uv sync --extra rgbd 2>/dev/null; then
        echo "✓ RGB-D dependencies installed"
        echo ""

        # Check if udev rules need to be installed
        if [ ! -f /etc/udev/rules.d/99-obsensor-libusb.rules ]; then
            echo "Note: Orbbec udev rules not found."
            echo "To use the Orbbec camera, you may need to install udev rules:"
            echo "  1. Clone pyorbbecsdk: git clone https://github.com/orbbec/pyorbbecsdk.git"
            echo "  2. Run: cd pyorbbecsdk/scripts && sudo ./install_udev_rules.sh"
            echo "  3. Reload rules: sudo udevadm control --reload-rules && sudo udevadm trigger"
            echo ""
        fi
    else
        echo "⚠ RGB-D dependencies not available, webcam mode will be used"
        echo ""
    fi
else
    echo "Note: RGB-D camera (pyorbbecsdk) is only available on Linux."
    echo "      Webcam mode will be used on $PLATFORM."
    echo ""
fi

# Download Pose Landmarker Model
if [ ! -f pose_landmarker.task ]; then
    echo "Downloading Pose Landmarker Model..."
    curl -o pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
    echo "✓ Pose Landmarker Model downloaded"
else
    echo "✓ Pose Landmarker Model already exists"
fi
echo ""

# Download Hand Landmarker Model
if [ ! -f hand_landmarker.task ]; then
    echo "Downloading Hand Landmarker Model..."
    curl -o hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    echo "✓ Hand Landmarker Model downloaded"
else
    echo "✓ Hand Landmarker Model already exists"
fi
echo ""

echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Run with mock robot (development):"
echo "  uv run griffin-teleop --mock --camera 0"
echo ""
echo "Run with real robot:"
echo "  uv run griffin-teleop --can can0 --camera 0"
echo ""
echo "Force webcam mode (disable RGB-D):"
echo "  uv run griffin-teleop --mock --camera 0 --no-depth"
echo ""
echo "See all options:"
echo "  uv run griffin-teleop --help"
echo ""
