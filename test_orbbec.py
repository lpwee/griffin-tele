#!/usr/bin/env python
"""
Minimal script to test pyorbbecsdk for RGB and Depth capture.

Mirrors the RealSense minimal test script pattern for Orbbec Gemini cameras.
Captures frames and saves them to disk, following the official example pattern.
"""

import pyorbbecsdk as ob
import numpy as np
import cv2
import os

# Configuration parameters (as variables, not flags)
DEVICE_INDEX = 0  # Set to device index (0, 1, 2, etc.), or None to use first device
WIDTH = 640
HEIGHT = 480
FPS = 30

# Initialize Orbbec context
context = ob.Context()
device_list = context.query_devices()

if device_list.get_count() == 0:
    raise RuntimeError("No Orbbec devices found")

# Select device
if DEVICE_INDEX is not None:
    if DEVICE_INDEX < 0 or DEVICE_INDEX >= device_list.get_count():
        raise RuntimeError(
            f"Device index {DEVICE_INDEX} is out of range. "
            f"Found {device_list.get_count()} device(s), valid indices are 0-{device_list.get_count()-1}."
        )
    device = device_list.get_device_by_index(DEVICE_INDEX)
else:
    device = device_list.get_device_by_index(0)

# Get device info
try:
    info = device.get_device_info()
    name = info.name() if hasattr(info, "name") else "Orbbec"
    serial = info.serial_number() if hasattr(info, "serial_number") else ""
except Exception:
    name, serial = "Orbbec", ""

print(f"Using device: {name}")
print(f"Serial number: {serial}")

# Create pipeline
pipeline = ob.Pipeline(device)

# Get stream profile lists
color_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
depth_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)

if color_profile_list is None or color_profile_list.get_count() == 0:
    raise RuntimeError("No color stream profiles available")

if depth_profile_list is None or depth_profile_list.get_count() == 0:
    raise RuntimeError("No depth stream profiles available")

# Try to get specific profiles matching requested parameters
color_profile = None
depth_profile = None

try:
    # Try MJPG format first for better bandwidth
    color_profile = color_profile_list.get_video_stream_profile(
        WIDTH, HEIGHT, ob.OBFormat.MJPG, FPS
    )
    print(f"Using MJPG color profile: {WIDTH}x{HEIGHT}@{FPS}fps")
except Exception:
    try:
        # Fallback to any format with matching resolution/fps
        color_profile = color_profile_list.get_video_stream_profile(
            WIDTH, 0, ob.OBFormat.ANY, FPS
        )
        print(f"Using ANY format color profile: {WIDTH}x*@{FPS}fps")
    except Exception:
        # Use default profile
        color_profile = color_profile_list.get_default_video_stream_profile()
        print("Using default color profile (requested profile not available)")

try:
    # Try Y12 format for depth (12-bit depth)
    depth_profile = depth_profile_list.get_video_stream_profile(
        WIDTH, HEIGHT, ob.OBFormat.Y12, FPS
    )
    print(f"Using Y12 depth profile: {WIDTH}x{HEIGHT}@{FPS}fps")
except Exception:
    try:
        # Fallback to Y11 format
        depth_profile = depth_profile_list.get_video_stream_profile(
            WIDTH, HEIGHT, ob.OBFormat.Y11, FPS
        )
        print(f"Using Y11 depth profile: {WIDTH}x{HEIGHT}@{FPS}fps")
    except Exception:
        # Use default profile
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        print("Using default depth profile (requested profile not available)")

# Configure streams
config = ob.Config()
config.enable_stream(color_profile)
config.enable_stream(depth_profile)

# Start pipeline
pipeline.start(config)

print(f"\nPipeline started successfully!")

# Get actual stream profiles to verify
color_video_profile = color_profile.as_video_stream_profile()
depth_video_profile = depth_profile.as_video_stream_profile()

print(f"\nActual stream profiles:")
print(f"  Color: {color_video_profile.get_width()}x{color_video_profile.get_height()}@{color_video_profile.get_fps()}fps, format={color_video_profile.get_format()}")
print(f"  Depth: {depth_video_profile.get_width()}x{depth_video_profile.get_height()}@{depth_video_profile.get_fps()}fps, format={depth_video_profile.get_format()}")

# Helper function to convert color frame to RGB
def frame_to_rgb_image(frame: ob.VideoFrame) -> np.ndarray:
    """Convert Orbbec frame to RGB format."""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    if color_format == ob.OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return image
    elif color_format == ob.OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == ob.OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUYV)
    elif color_format == ob.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # imdecode returns BGR, convert to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == ob.OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
    else:
        print(f"Warning: Unsupported color format: {color_format}, attempting default conversion")
        # Default: assume RGB layout
        return np.resize(data, (height, width, 3))

# Capture 30 frames to give autoexposure, etc. a chance to settle
print(f"\nWarming up (30 frames)...")
for i in range(30):
    frameset = pipeline.wait_for_frames(1000)
    if frameset is None:
        print(f"Warning: No frameset received during warmup frame {i}")

print(f"Warmup complete. Capturing frames...")

# Create output directory
output_dir = "outputs/test/orbbec"
os.makedirs(output_dir, exist_ok=True)

# Capture frames and save them
for frame_num in range(10):
    # Wait for frames
    frameset = pipeline.wait_for_frames(1000)
    if frameset is None:
        print(f"  Frame {frame_num}: No frameset received, skipping...")
        continue
    
    # Get color frame
    color_frame = frameset.get_color_frame()
    if color_frame is not None:
        width = color_frame.get_width()
        height = color_frame.get_height()
        frame_format = color_frame.get_format()
        
        print(f"  Frame {frame_num}, Color stream: "
              f"Size: {width}x{height}, Format: {frame_format}")
        
        # Convert to RGB
        rgb_image = frame_to_rgb_image(color_frame)
        
        # Save RGB image (convert to BGR for OpenCV imwrite)
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        rgb_filename = f"{output_dir}/ob-rgb-{frame_num}.png"
        cv2.imwrite(rgb_filename, rgb_bgr)
        print(f"    Saved RGB: {rgb_filename}")
    
    # Get depth frame
    depth_frame = frameset.get_depth_frame()
    if depth_frame is not None:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        depth_format = depth_frame.get_format()
        depth_scale = depth_frame.get_depth_scale()
        
        # Process depth data (typically uint16)
        # Depth scale converts raw pixel values to millimeters
        # e.g., if scale=1.0, raw values are already in mm; if scale=0.001, multiply to get mm
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        
        # Apply depth scale to convert to millimeters (float for precision)
        depth_data_mm = depth_data.astype(np.float32) * depth_scale
        
        # Get center point coordinates
        center_x = width // 2
        center_y = height // 2
        
        # Get distance at center point (in millimeters after scaling)
        center_distance_mm = depth_data_mm[center_y, center_x]
        center_distance_m = center_distance_mm / 1000.0  # Convert to meters
        
        # Convert back to uint16 for saving (preserve precision)
        depth_image = depth_data_mm.astype(np.uint16)
        
        print(f"  Frame {frame_num}, Depth stream: "
              f"Size: {width}x{height}, Format: {depth_format}, "
              f"Scale: {depth_scale}, Data shape: {depth_image.shape}, dtype: {depth_image.dtype}")
        print(f"    Center point ({center_x}, {center_y}) distance: "
              f"{center_distance_m:.3f} m ({center_distance_mm:.1f} mm) [raw pixel: {depth_data[center_y, center_x]}, scale: {depth_scale}]")
        
        # Save raw depth (16-bit PNG)
        raw_depth_filename = f"{output_dir}/ob-depth-raw-{frame_num}.png"
        cv2.imwrite(raw_depth_filename, depth_image)
        print(f"    Saved raw depth: {raw_depth_filename}")
        
        # Create colorized depth visualization (using OpenCV colormap)
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colorized_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        colorized_filename = f"{output_dir}/ob-depth-colorized-{frame_num}.png"
        cv2.imwrite(colorized_filename, colorized_depth)
        print(f"    Saved colorized depth: {colorized_filename}")
        
        # Also save custom RGB-encoded depth (matching image_writer.py encoding)
        # Encode 16-bit depth into RGB channels to preserve precision:
        # - R channel = high byte (upper 8 bits)
        # - G channel = low byte (lower 8 bits)
        # - B channel = 0 (unused)
        # This is lossless encoding that preserves full 16-bit precision
        if depth_image.dtype == np.uint16:
            high_byte = (depth_image >> 8).astype(np.uint8)  # Upper 8 bits
            low_byte = (depth_image & 0xFF).astype(np.uint8)  # Lower 8 bits
            zero_channel = np.zeros_like(high_byte, dtype=np.uint8)  # B channel = 0
            # Stack as RGB: (H, W, 3)
            rgb_encoded_depth = np.stack([high_byte, low_byte, zero_channel], axis=-1)
            custom_rgb_filename = f"{output_dir}/ob-depth-rgb-encoded-{frame_num}.png"
            cv2.imwrite(custom_rgb_filename, rgb_encoded_depth)
            print(f"    Saved RGB-encoded depth: {custom_rgb_filename}")
        else:
            print(f"    Warning: Depth dtype is {depth_image.dtype}, expected uint16 for RGB encoding")

# Stop pipeline
pipeline.stop()
print(f"\nPipeline stopped. Test complete!")
