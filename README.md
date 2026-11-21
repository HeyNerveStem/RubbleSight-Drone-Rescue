# RubbleSight – Post-Disaster Search and Rescue (Vision Baseline)

Public companion repository for my M.Tech dissertation at NSUT, New Delhi:
*"An Edge-AI Enabled UAV Framework Fusing Vision and RF Sensing for Detection of Visible and Buried Disaster Victims"*

This repo contains the **line-of-sight vision baseline**: person detection on aerial (UAV) imagery using YOLO.
The full system, including RF sensing and fusion, is developed in a private repository and will be released after evaluation.

The problem framing builds on PDSR, a post-disaster search-and-rescue UAV framework by Abdellatif et al. (IEEE IoT Magazine, 2025).

## Quick start
​bash
pip install -r requirements.txt
python src/vision/baseline_detect.py --source path/to/drone_image.jpg --save

Output: the number of persons detected per frame, throughput (FPS), and annotated images under `runs/detect/`.

## Structure
- `src/vision/` – detection code
- `docs/roadmap.md` – project phases and status

## Tech stack
Python · PyTorch · Ultralytics YOLO · OpenCV

## Author
Yashdeep, M.Tech IT, NSUT
