# Object Detection & Tracking (YOLOv8 + Deep SORT)

Real-time object detection and multi-object tracking using a webcam or video file.

## Requirements
- Python 3.9+
- Webcam (or a video file)

## Install
```bash
pip install ultralytics opencv-python deep-sort-realtime
```

## Run
```bash
# Webcam (default: camera 0)
python detect_track.py

# Video file
python detect_track.py --source path/to/video.mp4

# Save output
python detect_track.py --source video.mp4 --save output.mp4
```

Press `q` to quit the preview window.

## What it does
1. Opens webcam / video via OpenCV.
2. Runs YOLOv8 (pre-trained on COCO, 80 classes) on every frame.
3. Feeds detections to Deep SORT for identity-preserving tracking.
4. Draws bounding boxes with `ClassName ID:<track_id> <conf>` labels.
5. Shows live FPS.
