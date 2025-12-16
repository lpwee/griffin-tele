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



### Things to think about:
1. Pose Estimator Shoulder will be the base(?)origin(?) of the URDF 
2. Wrist Position is the end-effector's target position


New ranges for relative coordinates (estimated from arm proportions):

 Assuming:
 - Typical arm length (shoulder to wrist): ~60cm
 - Camera view width: ~200cm at operating distance
 - Normalized arm reach: ~0.3 (60/200)

 Relative wrist position (wrist - shoulder) typical ranges:
 - X (left/right): arm reaches ~0.25 either side of shoulder → (-0.25, 0.25)
 - Y (up/down): wrist typically below shoulder (positive Y in camera = down) → (-0.05, 0.30)
 - Z (depth): arm extends forward ~0.2 from shoulder plane → (-0.15, 0.20)

```
 uv run griffin-teleop --mock --camera 1 --show-arm-viz --verbose-ik --fps 11
```

new TODO:
[ ] measure latency
[ ] fix orientation detection (see teleop TODO)
[ ] gripper open close
