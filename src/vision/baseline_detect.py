"""
RubbleSight - Vision baseline
-----------------------------
Baseline person detection on aerial (UAV) images or video using a
pretrained YOLO model. This is the line-of-sight (LOS) component for
visible victims; buried victims are handled by the separate
non-line-of-sight (NLOS) RF sensing module (private, in development).

Usage:
    python src/vision/baseline_detect.py --source path/to/image_or_video
    python src/vision/baseline_detect.py --source 0          # webcam
    python src/vision/baseline_detect.py --source img.jpg --conf 0.3 --save
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

PERSON_CLASS_ID = 0  # "person" in the COCO dataset


def parse_args():
    p = argparse.ArgumentParser(description="RubbleSight vision baseline")
    p.add_argument("--source", required=True,
                   help="Image, video, folder, or webcam index (0)")
    p.add_argument("--weights", default="yolov8n.pt",
                   help="YOLO weights (default: yolov8n.pt, downloads automatically)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size")
    p.add_argument("--save", action="store_true",
                   help="Save annotated output to runs/detect/")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    start = time.perf_counter()
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=[PERSON_CLASS_ID],
        save=args.save,
        stream=True,
        verbose=False,
    )

    frames, total_people = 0, 0
    for r in results:
        frames += 1
        n = len(r.boxes)
        total_people += n
        name = Path(r.path).name if r.path else f"frame_{frames}"
        print(f"[{name}] persons detected: {n}")

    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    print("-" * 40)
    print(f"Frames processed : {frames}")
    print(f"Total detections : {total_people}")
    print(f"Throughput       : {fps:.2f} FPS")
    if args.save:
        print("Annotated output saved under runs/detect/")


if __name__ == "__main__":
    main()
