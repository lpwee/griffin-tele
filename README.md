# griffin-tele

Install uv:
```
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up environment, run.
```
uv sync
uv run griffin-teleop --mock --camera 1 --no-gripper
```


### MediaPipe Pose Detection Model

Download the Pose Detection Model
Ref: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker (Three sizes, lite, full, heavy)
```
curl -o pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```


Download the Hands Detection Model for Pinchers
Ref : https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
```
curl -o hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```


## Planning
- [ ] RGB Camera for now, do we want RGB-D?
- [x] Get Pose Detection Model working
- [x] Get Hand Detection Model working
- [x] Grab URDF files from the [piper_ros repository](https://github.com/agilexrobotics/piper_ros?tab=readme-ov-file#0-%E6%B3%A8%E6%84%8Furdf%E7%89%88%E6%9C%AC)

current version >= S-V1.6-3	piper_description.urdf
- [x] Explore IKPy or PyBullet for solving IK problems
- [ ] Or integrate with MoveIt, will resolve collision and IK

## new TODO:
- [x] measure latency
- [ ] fix orientation detection
- [ ] gripper open close


### Things to think about:
1. Pose Estimator Shoulder will be the base(?)origin(?) of the URDF 
2. Wrist Position is the end-effector's target position

3. crazy idea: IK to solve for first 5 joints, hand orientation solved by joint6
4. jerkiness when selecting random solutions from IK

Possible approaches to Z coordinate problem
- [ ] IMU
- [ ] MoCap Balls
- [ ] DepthAnything
- [ ] RGB-D


RGB ONLY ROUTE
1. Mediapipe output -> Pixels (based on image width)
2. Pixels to real world(arm) opencv calibration(chessboard thingy)
3. Derive the intrinsics (which is a multiplier from pixel space to real world space)

