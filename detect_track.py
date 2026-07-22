"""
Real-time object detection + tracking.
- Detector: YOLOv8 (Ultralytics, pretrained on COCO)
- Tracker : Deep SORT (deep-sort-realtime)
- I/O     : OpenCV (webcam or video file)

Usage:
    python detect_track.py                       # webcam 0
    python detect_track.py --source video.mp4    # video file
    python detect_track.py --save out.mp4        # save annotated video
"""
import argparse
import time

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="Webcam index (e.g. 0) or path to video file.")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLOv8 weights (auto-downloaded on first run).")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Detection confidence threshold.")
    p.add_argument("--save", default=None,
                   help="Optional output video path (e.g. out.mp4).")
    return p.parse_args()


def open_source(src: str):
    # numeric string -> webcam index
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")
    return cap


def color_for_id(track_id: int):
    # Deterministic pseudo-random color per track ID
    tid = int(track_id)
    return (
        (tid * 37) % 256,
        (tid * 91) % 256,
        (tid * 151) % 256,
    )


def main():
    args = parse_args()

    print("[info] loading YOLO model:", args.model)
    model = YOLO(args.model)
    class_names = model.names

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        embedder="mobilenet",
        half=True,
    )

    cap = open_source(args.source)

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"[info] saving output to {args.save}")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- Detection ---
        results = model.predict(frame, conf=args.conf, verbose=False)[0]

        # Build detections in Deep SORT format: ([x, y, w, h], confidence, class)
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], conf, class_names[cls_id])
            )

        # --- Tracking ---
        tracks = tracker.update_tracks(detections, frame=frame)

        # --- Draw ---
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            l, top, r, b = map(int, t.to_ltrb())
            label = t.get_det_class() or "object"
            color = color_for_id(tid)

            cv2.rectangle(frame, (l, top), (r, b), color, 2)
            caption = f"{label} ID:{tid}"
            (tw, th), _ = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (l, top - th - 8), (l + tw + 6, top), color, -1)
            cv2.putText(
                frame, caption, (l + 3, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        if writer is not None:
            writer.write(frame)

        cv2.imshow("YOLOv8 + Deep SORT", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
