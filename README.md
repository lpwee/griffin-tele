# griffin-tele

Install uv:
```
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up environment
```
uv sync
uv run <python file>
```


### MediaPipe Pose Detection Model

Download the Pose Detection Model
Ref: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
```
curl -o pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

Download the Hands Detection Model for Pinchers
Ref : https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
```
curl -o hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```


## Planning
[ ] RGB Camera for now, do we want RGB-D?
[ ] Get Pose Detection Model working
[ ] Get Hand Detection Model working
[ ] Grab URDF files from the (https://github.com/agilexrobotics/piper_ros?tab=readme-ov-file#0-%E6%B3%A8%E6%84%8Furdf%E7%89%88%E6%9C%AC)[piper_ros repository]

current version >= S-V1.6-3	piper_description.urdf
[ ] Explore IKPy or PyBullet for solving IK problems
[ ] Or integrate with MoveIt, will resolve collision and IK

## first-build
  Run with:
  ### Mock robot (no hardware)
  uv run griffin-teleop --mock --camera 0

  ### Real robot
  uv run griffin-teleop --can can0 --camera 0