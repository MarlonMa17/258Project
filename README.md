##  Task distribution among group members

- Jinwoo Bae
Identified the fundamental issue for intelligent traffic control and explained the real-world application setting. chosen and examined the UA-DETRAC dataset, paying particular attention to its environmental conditions, labeled classifications, and structure. assessed the dataset's suitability for object detection model training and testing by looking at important features such vehicle types, occlusion levels, truncation, lighting conditions, and object scales. summarized the results of the dataset interpretation and provided assistance for the evaluation discussion in the project report.

- Honghao Ma
Pipeline Engineering & Evaluation: Led the complete data preprocessing workflow, including frame extraction, annotation conversion, and dataset splitting tailored for optimized model input. Carried out localized training and validation across multiple object detection architectures (e.g., YOLOv5n, YOLOv5m), iteratively tuning hyperparameters for performance benchmarking under constrained resources.
Integration & Deployment: Implemented the end-to-end inference pipeline for video processing, integrating frame-wise detection with visual annotation overlay and result output for performance analysis. Developed scripts enabling efficient model evaluation on test video sequences with support for batch visualization.
Project Infrastructure: Set up and maintained the GitHub repository, ensuring reproducibility through organized codebase structure and detailed README documentation. Authored significant portions of the final project report and presentation slides, particularly focusing on experimental evaluation metrics, visual result comparisons, and model performance discussion across different detection settings.
https://github.com/MarlonMa17/258Project

- Itzel Xu
Literature & Model Research:Conducted an in-depth review of key object detection papers. Researched model architecture details and compared various YOLO versions and their practical implications in constrained edge deployment scenarios.
Dataset Evaluation: Validated the UA-DETRAC dataset, ensuring proper label mapping and quality checks for robust traffic detection.
Deployment & Debugging: Resolved configuration issues in both training and inference pipelines (e.g., module imports, device configuration, input tensor shape conversion, normalization, and augmentation) within the Kaggle environment.
Collaborated with team members contributing to clear project documentation and presentation material.


---

# Smart City Vehicle Detection with YOLOv5

This project implements a vehicle detection and classification system using YOLOv5, trained on the [UA-DETRAC](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset) dataset. The goal is to enable accurate recognition of vehicles (bus, car, van, truck) in traffic videos for smart city applications.

---

## Project Structure

```
258project/
├── yolov5/                # YOLOv5 cloned repo (excluded from GitHub)
├── content/               # Dataset (excluded from GitHub)
├── runs/                  # YOLOv5 outputs: training, validation, inference
├── videotraffic/          # Input traffic videos for inference (excluded from GitHub)
├── runs/detrac.yaml       # Dataset YAML configuration
├── smartcity.ipynb        # Main notebook for training & inference
└── README.md
```

---

## Dataset: UA-DETRAC

- Link: [UA-DETRAC on Kaggle](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset)
- You must manually download and extract the dataset into `content/`
- Label format (confirmed):
  - `0`: bus
  - `1`: car
  - `2`: van
  - `3`: truck

---

##  Installation

```bash
pip install -r requirements.txt
```

---

##  Training

```bash
python yolov5/train.py --img 640 --batch 32 --epochs 5 --data runs/detrac.yaml --weights yolov5m.pt --device 0 --cache
```

Training results are saved to: `yolov5/runs/train/exp*/`

---

##  Validation

```bash
python yolov5/val.py --weights yolov5/runs/train/exp9/weights/best.pt --data runs/detrac.yaml --img 416
```

Evaluation results are saved to: `yolov5/runs/val/exp*/`

---

##  Inference on Video

Run inference inside the notebook using:

```python
smartcity.ipynb
```

Results will be saved to: `yolov5/runs/inference/`

---



##  Output/Result in runs/Directory

- `runs/train/exp*`: Contains training outputs like `best.pt`, loss curves, and sample images.
- `runs/val/exp*`: Stores evaluation results such as mAP curves and confusion matrices.
- `runs/inference/`: Contains annotated traffic videos generated from inference using the trained model.

---

##  Key references

- L. Wen et al., “UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking,” Computer Vision and Image Understanding, vol. 193, p. 102907, Apr. 2020, doi: https://doi.org/10.1016/j.cviu.2020.102907.
- https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset/code
-  Khanam, R., & Hussain, M. (2024). What is YOLOv5: A deep look into the internal features of the popular object detector [Preprint]. arXiv. https://arxiv.org/abs/2407.2089
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real‑time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 779–788). https://doi.org/10.1109/CVPR.2016.91
- torchvision.ops.nms. (n.d.). In PyTorch Documentation. Retrieved May 11, 2025, from https://pytorch.org/vision/stable/ops.html#torchvision.ops.nms

