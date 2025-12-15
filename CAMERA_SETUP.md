# Camera Setup for WSL

This guide explains how to use a Windows camera with your WSL-based teleoperation system.

## Overview

Since WSL doesn't have direct access to hardware devices like cameras, we use a client-server approach:
- **Windows**: Runs a camera server that captures frames and streams them over HTTP
- **WSL**: Connects to the stream via network

## Setup Instructions

### Step 1: Install Dependencies on Windows

Open **PowerShell** or **Command Prompt** on Windows and install the required Python packages:

```bash
pip install opencv-python flask
```

### Step 2: Start the Camera Server on Windows

1. Navigate to your project directory in Windows (the same directory accessible from WSL):
   ```bash
   cd \\wsl$\Ubuntu\home\pingw\projects\griffin-tele
   ```

2. Run the camera server:
   ```bash
   python windows_camera_server.py --camera 0 --port 5000
   ```

   Parameters:
   - `--camera`: Camera device ID (usually 0 for default webcam, 1 for external)
   - `--port`: Server port (default: 5000)
   - `--host`: Server host (default: 0.0.0.0 to allow WSL connections)

3. The server will print the stream URL. Note it down.

### Step 3: Find Your Windows IP Address

You need to know your Windows machine's IP address to connect from WSL.

**Option A: Using hostname (easiest)**
```bash
# On WSL, run:
ping $(hostname).local
```

**Option B: Find IP manually**

On Windows PowerShell:
```powershell
ipconfig
```

Look for the "Ethernet adapter" or "Wireless LAN adapter" section and note the **IPv4 Address** (e.g., `172.20.160.1`).

### Step 4: Test the Stream from WSL

From WSL, test that you can receive the stream:

```bash
python -m src.camera_capture "http://<WINDOWS-IP>:5000/video_feed" --duration 10
```

Replace `<WINDOWS-IP>` with your Windows IP address (e.g., `172.20.160.1`).

Example:
```bash
python -m src.camera_capture "http://172.20.160.1:5000/video_feed" --duration 10
```

This will display the stream for 10 seconds and show FPS statistics.

### Step 5: Run Teleoperation with Network Stream

Now you can run your teleoperation system using the network stream:

```bash
python -m src.teleop --camera "http://172.20.160.1:5000/video_feed" --mock
```

Replace the IP address with yours, and remove `--mock` when using real hardware.

## Quick Reference

### Start Windows Server
```bash
# On Windows
python windows_camera_server.py --camera 0 --port 5000
```

### Run Teleop from WSL
```bash
# On WSL
python -m src.teleop --camera "http://<WINDOWS-IP>:5000/video_feed" --mock
```

## Troubleshooting

### Camera Not Opening on Windows
- Make sure no other application is using the camera (Teams, Zoom, etc.)
- Try different camera IDs: `--camera 1`, `--camera 2`, etc.
- On Windows, camera ID 0 is usually the built-in webcam

### Cannot Connect from WSL
- Check Windows Firewall settings - you may need to allow Python/Flask
- Verify the IP address is correct
- Try using `localhost` if running on the same machine
- Make sure the server is running and shows "Server ready!"

### Low FPS or Lag
- Reduce camera resolution in `windows_camera_server.py` (lines 29-30)
- Use wired Ethernet connection instead of WiFi for better performance
- Adjust JPEG quality (line 56): lower value = less quality but faster

### Stream Freezes or Drops Frames
- This is normal with network streams - the code will continue with the latest available frame
- Check network stability
- Consider using a local camera if performance is critical

## Alternative: Use Local Camera (if available)

If you have USB passthrough or are using WSL1, you might be able to use the camera directly:

```bash
python -m src.teleop --camera 0 --mock
```

This will work if `/dev/video0` exists and is accessible from WSL.

## Advanced: Using Environment Variables

You can set a default stream URL in your shell profile:

```bash
# Add to ~/.bashrc
export CAMERA_STREAM_URL="http://172.20.160.1:5000/video_feed"

# Then run with:
python -m src.teleop --camera "$CAMERA_STREAM_URL" --mock
```
