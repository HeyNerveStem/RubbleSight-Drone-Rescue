# Roadmap

RubbleSight detects disaster victims from a UAV using two complementary sensing paths:
**line-of-sight (LOS)** vision for visible victims, and **non-line-of-sight (NLOS)** RF sensing for victims buried under debris.

| Phase | Work | Status |
|---|---|---|
| Phase I | Literature review, problem definition, system design | ✅ Completed (Sept 2026) |
| LOS baseline | Pretrained YOLO person detection on aerial imagery | ✅ Done |
| LOS baseline evaluation | Precision/recall on VisDrone2019-DET val (548 images, 13,969 people) | ✅ Done (Nov 2026) |
| LOS on disaster data | Baseline on real Turkey 2023 earthquake footage (D-RespNeT, 75 images, 579 people) | ✅ Done (Nov 2026) |
| LOS improvement | Fine-tuning on disaster imagery (D-RespNeT train split) and thermal data, small-object detection | ⏳ Next |
| NLOS dataset | RS-NLOS v1.0: synthetic radar + Wi-Fi dataset with Sionna RT ray tracing ([summary](rs-nlos.md)) | ✅ Built (Oct 2026) |
| NLOS detection | Breathing detection under drone motion, Wi-Fi phone detection | 🔒 In development (private) |
| NLOS validation | Testing on real UWB radar recordings | ⏳ Planned |
| Fusion and edge deployment | Combined LOS + NLOS victim map, model optimisation for onboard hardware | 🔒 In development (private) |
| Phase II | Full evaluation, dissertation and paper | ⏳ Planned |

🔒 = part of the dissertation's novel contribution; released after evaluation and publication.

## Repository structure

```
RubbleSight-Drone-Rescue/
├── src/vision/        # LOS detection and evaluation code
├── results/           # Evaluation metrics (JSON)
├── assets/            # Example detection images
├── docs/              # Roadmap and design notes
└── requirements.txt
```
