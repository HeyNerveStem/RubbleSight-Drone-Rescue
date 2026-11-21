"""
RubbleSight - Vision baseline evaluation on real earthquake imagery
-------------------------------------------------------------------
Measures how well a pretrained (COCO) YOLO model detects people in drone
footage from the 2023 Turkey earthquake, using the D-RespNeT dataset
(Cranfield University, Roboflow export, MIT licence) as ground truth.

D-RespNeT is labelled with instance-segmentation polygons for 28 disaster
classes. For this evaluation:
  - "civilian_visible" and "rescue_team" are individual people -> ground truth
  - "group_of_civilians" marks a crowd as one region -> ignored (detections
    inside it are neither rewarded nor penalised)
  - every other class is not a person and is not used
Polygons are converted to bounding boxes. A detection is correct at IoU >= 0.5.

Usage:
    python src/vision/eval_respnet.py --data "path/to/D_RespNeT - Instance Segmentation.v7i.yolov8" \
        --splits valid test --weights yolov8n.pt --imgsz 1280 --out results/respnet_yolov8n_1280.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from eval_visdrone import inside_ignored, match

PERSON_CLASS_ID = 0  # COCO "person"
CLASSES = ['backhoe_loader', 'bridge', 'building_collapsed', 'building_damaged', 'building_undamaged', 'bus',
           'car_damaged', 'car_undamaged', 'civilian_visible', 'common_dump_truck', 'crawler_crane',
           'crawler_loader', 'debris_heavy', 'debris_light', 'debris_moderate', 'drone_landing_zone_safe',
           'entry_door_accessible', 'entry_door_blocked', 'entry_gap_accessible', 'entry_gap_block_accessible',
           'entry_window_accessible', 'entry_window_blocked', 'excavator', 'group_of_civilians', 'rescue_team',
           'road', 'rubble', 'truck']
PERSON_CLASSES = {CLASSES.index("civilian_visible"), CLASSES.index("rescue_team")}
IGNORE_CLASSES = {CLASSES.index("group_of_civilians")}


def load_gt(label_path: Path, w: int, h: int):
    people, ignore = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            k = int(v[0])
            pts = np.array(v[1:], dtype=float).reshape(-1, 2)
            box = [pts[:, 0].min() * w, pts[:, 1].min() * h, pts[:, 0].max() * w, pts[:, 1].max() * h]
            if k in PERSON_CLASSES:
                people.append(box)
            elif k in IGNORE_CLASSES:
                ignore.append(box)
    return np.array(people, dtype=float).reshape(-1, 4), np.array(ignore, dtype=float).reshape(-1, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="D-RespNeT YOLOv8 export folder")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model = YOLO(args.weights)
    root = Path(args.data)
    tp = n_det = n_gt = n_img = 0
    t_infer = 0.0
    for split in args.splits:
        for img in sorted((root / split / "images").glob("*.jpg")):
            r_t0 = time.perf_counter()
            r = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                              classes=[PERSON_CLASS_ID], verbose=False)[0]
            t_infer += time.perf_counter() - r_t0
            h, w = r.orig_shape
            gt, ignore = load_gt(root / split / "labels" / (img.stem + ".txt"), w, h)
            b = r.boxes
            det = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
            conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
            det = det[np.argsort(-conf)]
            det = det[~inside_ignored(det, ignore)]
            tp += match(det, gt)
            n_det += len(det)
            n_gt += len(gt)
            n_img += 1

    precision = tp / n_det if n_det else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "dataset": "D-RespNeT (Turkey earthquake 2023)", "splits": args.splits,
        "weights": args.weights, "imgsz": args.imgsz, "conf": args.conf,
        "images": n_img, "gt_persons": n_gt, "detections": n_det, "true_positives": tp,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "ms_per_image": round(1000 * t_infer / max(1, n_img), 1),
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
