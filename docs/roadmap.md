# Roadmap

| Phase | Work | Status |
|---|---|---|
| Phase I | Literature review, problem definition, system design | ✅ Completed (Sept 2026) |
| Vision baseline | Pretrained YOLO person detection on aerial imagery | ✅ Done |
| Baseline evaluation | Precision/recall on VisDrone2019-DET val (548 images) | ✅ Done (Nov 2026) |
| Vision module | Fine-tuning on aerial / thermal datasets, small-object detection | ⏳ Next |
| RF sensing module | Non-line-of-sight detection of buried victims | ⏳ Planned |
| Fusion and edge deployment | Combined victim-location output, model optimisation | ⏳ Planned |
| Phase II | Evaluation, dissertation and paper | ⏳ Planned |

## Repository structure

```
RubbleSight/
├── src/vision/        # Detection and evaluation code
├── results/           # Evaluation metrics (JSON)
├── assets/            # Example detection images
├── docs/              # Roadmap and design notes
└── requirements.txt
```
