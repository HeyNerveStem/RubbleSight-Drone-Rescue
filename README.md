# RubbleSight – Post-Disaster Search and Rescue (Vision Baseline)

Public companion repository for my M.Tech dissertation at NSUT, New Delhi:
*"An Edge-AI Enabled UAV Framework Fusing Vision and RF Sensing for Detection of Visible and Buried Disaster Victims"*

This repo contains the **line-of-sight vision baseline**: person detection on aerial (UAV) imagery using YOLO.
The full system, including RF sensing and fusion, is developed in a private repository and will be released after evaluation.

The problem framing builds on PDSR, a post-disaster search-and-rescue UAV framework by Abdellatif et al. (IEEE IoT Magazine, 2025).

## Baseline results on real drone imagery

I evaluated off-the-shelf YOLOv8 models (pretrained on COCO, **no fine-tuning**) on the full
[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) validation set:
548 drone images containing 13,969 labelled people (pedestrian + people classes).
A detection counts as correct at IoU ≥ 0.5, confidence threshold 0.25.

| Model | Input size | Precision | Recall | F1 | CPU time / image |
|---|---|---|---|---|---|
| YOLOv8n | 640 | 0.801 | 0.073 | 0.134 | 70 ms |
| YOLOv8n | 1280 | 0.848 | 0.213 | 0.340 | 223 ms |
| YOLOv8s | 640 | 0.829 | 0.121 | 0.211 | 160 ms |
| YOLOv8s | 1280 | **0.858** | **0.266** | **0.406** | 598 ms |

*Timing measured on a 2-core CPU without a GPU; edge hardware with acceleration will be much faster.*

**What this shows**
- **Detections are reliable but most people are missed.** Precision stays above 80%, but even the best setup finds only about 27% of people. From drone altitude, people are only a few pixels tall, and COCO-trained models rarely see targets that small.
- **Input resolution matters more than model size.** Going from 640 to 1280 px roughly triples recall for YOLOv8n (7.3% → 21.3%), a bigger gain than switching to the larger YOLOv8s at 640.
- **There is a speed/accuracy trade-off for edge deployment.** Higher resolution and bigger models cost about 3–9× more compute per image.

These results motivate the next stages: fine-tuning on aerial and thermal data, small-object techniques such as tiled inference, and adding non-line-of-sight sensing for victims the camera cannot see at all.

### Example detections (left: 640 px, right: 1280 px, YOLOv8n)

![Comparison 1](assets/compare_1.jpg)
![Comparison 2](assets/compare_2.jpg)
![Comparison 3](assets/compare_3.jpg)

## Quick start

```bash
pip install -r requirements.txt

# Detect people in an image, video or folder
python src/vision/baseline_detect.py --source path/to/drone_image.jpg --save

# Reproduce the evaluation (download VisDrone2019-DET-val first)
python src/vision/eval_visdrone.py --data path/to/VisDrone2019-DET-val \
    --weights yolov8n.pt --imgsz 1280 --out results/yolov8n_1280.json
```

Raw metrics for each run are in [`results/`](results/).

## Structure
- `src/vision/baseline_detect.py` – person detection on images, video or webcam
- `src/vision/eval_visdrone.py` – precision/recall evaluation against VisDrone ground truth
- `results/` – metrics from each evaluation run
- `assets/` – example detection images
- `docs/roadmap.md` – project phases and status

## Tech stack
Python · PyTorch · Ultralytics YOLO · OpenCV · NumPy

## Author
Yashdeep, M.Tech IT, NSUT
