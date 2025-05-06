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

##  Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [UA-DETRAC Dataset](https://detrac-db.rit.albany.edu/) by CASIA
