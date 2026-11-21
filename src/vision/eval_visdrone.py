"""
RubbleSight - Vision baseline evaluation on VisDrone
----------------------------------------------------
Measures how well a pretrained (COCO) YOLO model detects people in real
drone imagery, using the VisDrone2019-DET validation set as ground truth.

VisDrone shows urban scenes, not disaster sites. It is used as a first
benchmark for the core line-of-sight (LOS) challenge: tiny people seen from
altitude. Disaster-scene evaluation is a later stage; the same script works
on any dataset converted to the VisDrone annotation format.

VisDrone categories 1 (pedestrian) and 2 (people) are treated as "person".
Regions with score 0 ("ignored regions") are excluded. A detection counts
as correct when IoU >= 0.5 with an unmatched ground-truth person box.

Usage:
    python src/vision/eval_visdrone.py --data path/to/VisDrone2019-DET-val \
        --weights yolov8n.pt --imgsz 640 --out results/yolov8n_640.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0           # COCO "person"
VISDRONE_PERSON_CATS = {1, 2}  # pedestrian, people


def load_gt(ann_path: Path):
    boxes, ignore = [], []
    for line in ann_path.read_text().strip().splitlines():
        v = [int(x) for x in line.strip().strip(",").split(",")[:8]]
        x, y, w, h, score, cat = v[:6]
        box = [x, y, x + w, y + h]
        if score == 0 or cat == 0:
            ignore.append(box)
        elif cat in VISDRONE_PERSON_CATS:
            boxes.append(box)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(ignore, dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def inside_ignored(det: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """True for detections whose centre falls inside an ignored region."""
    if len(det) == 0 or len(ignore) == 0:
        return np.zeros(len(det), dtype=bool)
    cx = (det[:, 0] + det[:, 2]) / 2
    cy = (det[:, 1] + det[:, 3]) / 2
    return ((cx[:, None] >= ignore[None, :, 0]) & (cx[:, None] <= ignore[None, :, 2]) &
            (cy[:, None] >= ignore[None, :, 1]) & (cy[:, None] <= ignore[None, :, 3])).any(axis=1)


def match(det: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> int:
    """Greedy matching of score-sorted detections to ground truth; returns true positives."""
    if len(det) == 0 or len(gt) == 0:
        return 0
    ious = iou_matrix(det, gt)
    used, tp = set(), 0
    for i in range(len(det)):
        order = np.argsort(-ious[i])
        for j in order:
            if ious[i, j] < thr:
                break
            if j not in used:
                used.add(j)
                tp += 1
                break
    return tp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="VisDrone2019-DET-val folder (images/ + annotations/)")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images (0 = all)")
    p.add_argument("--out", default=None, help="Write metrics JSON here")
    args = p.parse_args()

    root = Path(args.data)
    images = sorted((root / "images").glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    model = YOLO(args.weights)

    tp = n_det = n_gt = 0
    t_infer = 0.0
    for img in images:
        gt, ignore = load_gt(root / "annotations" / (img.stem + ".txt"))
        t0 = time.perf_counter()
        r = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                          classes=[PERSON_CLASS_ID], verbose=False)[0]
        t_infer += time.perf_counter() - t0
        b = r.boxes
        det = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        det = det[np.argsort(-conf)]
        det = det[~inside_ignored(det, ignore)]
        tp += match(det, gt)
        n_det += len(det)
        n_gt += len(gt)

    precision = tp / n_det if n_det else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "weights": args.weights, "imgsz": args.imgsz, "conf": args.conf,
        "images": len(images), "gt_persons": n_gt, "detections": n_det, "true_positives": tp,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "ms_per_image": round(1000 * t_infer / max(1, len(images)), 1),
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
