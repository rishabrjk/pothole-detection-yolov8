# 🕳️ Real-Time Pothole Detection & Segmentation using YOLOv8

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=flat-square&logo=opencv)
![Roboflow](https://img.shields.io/badge/Roboflow-Annotated-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **Final Year B.Tech Major Project — MGIT Hyderabad (2024)**
> An end-to-end real-time pothole segmentation system using YOLOv8, trained on 2,023 annotated road images and deployed for live video inference using contour-based polygon masking.

---

## 📌 Project Overview

Road potholes are one of the leading causes of vehicle accidents and infrastructure damage across India and globally. Traditional road inspections rely on manual surveys — slow, costly, and inconsistent.

This project builds an **automated pothole detection and segmentation pipeline** using YOLOv8's instance segmentation model. Unlike basic bounding-box detection, this system draws precise **polygon contours** around each pothole in real time — giving maintenance teams not just a location, but an accurate shape and boundary of every defect detected.

> Built as a Major Project for the B.Tech in Electronics & Communication Engineering at Mahatma Gandhi Institute of Technology (MGIT), Hyderabad — 2024.

---

## 🚧 Problem Statement

| Challenge | Impact |
|---|---|
| Manual road inspection | Labour-intensive, infrequent, and expensive |
| Detection at scale | Thousands of km of road cannot be monitored manually |
| Reaction time | Damage often goes unreported until it becomes dangerous |
| Precision | Bounding boxes alone don't capture pothole shape or severity |

**Goal:** Build a system that can automatically detect and precisely segment potholes from video footage in real time — providing road authorities with both location and shape data for efficient maintenance prioritisation.

---

## 🧠 Approach & Methodology

```
Data Collection → Annotation (Roboflow) → Preprocessing → YOLOv8 Training → Evaluation → Real-Time Video Inference
```

### 1. Data Collection
- Sourced road images and video footage from **Google** and **YouTube** covering urban streets, highways, and rural roads across varied lighting and weather conditions
- Ensured diversity in pothole type, size, road surface, and environment

### 2. Data Annotation with Roboflow
- Annotated **2,023 road images** using **Roboflow**'s segmentation annotation tool
- Each pothole instance labelled with a **polygon segmentation mask** (not just a bounding box) — enabling precise boundary-level detection
- Dataset exported in YOLOv8 segmentation format (`yolov8`)

### 3. Data Preprocessing & Augmentation
Applied via Roboflow:
- Image resizing to 640×640 resolution
- Horizontal flipping
- Colour jitter and brightness variation
- Mosaic augmentation for improved generalisation

### 4. Model — YOLOv8 Segmentation
- Used **YOLOv8s-seg** (small segmentation variant) pre-trained on COCO, fine-tuned on the pothole dataset
- Architecture: CSPDarknet backbone → PANet neck → anchor-free detection head with segmentation masks
- Training: 100 epochs, image size 640, GPU-accelerated

### 5. Real-Time Inference Pipeline
- Loaded trained model (`best.pt`) using the Ultralytics API
- Processed live video frame-by-frame using **OpenCV**
- Drew **polygon contours** (not bounding boxes) around detected potholes using `cv2.polylines`
- Labelled each detection with class name and confidence score

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| Object Detection / Segmentation | YOLOv8 (Ultralytics) |
| Deep Learning Framework | PyTorch |
| Video & Image Processing | OpenCV |
| Data Annotation | Roboflow |
| Data Handling | NumPy, Pandas |
| Visualisation | Matplotlib |
| Environment | Jupyter Notebook / Google Colab |
| Version Control | Git & GitHub |

---

## 📊 Results

| Metric | Value |
|---|---|
| **Dataset Size** | **2,023 annotated images** |
| **Annotation Type** | Polygon segmentation masks (via Roboflow) |
| **Model** | YOLOv8s-seg (fine-tuned) |
| **Training Epochs** | 100 |
| **Inference Speed** | Real-time on standard GPU |
| **Evaluation Metrics** | Precision, Recall, mAP@0.5, mAP@0.5:0.95, IoU |

### Training Curves (Summary)
Training loss (box, cls, dfl) showed consistent decrease across 100 epochs. Precision and Recall metrics improved steadily. mAP@0.5 reached a strong plateau by epoch 80+, demonstrating successful convergence without overfitting.

### What the metrics mean (plain English)
- **Precision** — of every pothole the model flagged, what proportion was a real pothole
- **Recall** — of every real pothole in the image, what proportion did the model catch
- **mAP@0.5** — overall detection quality score at 50% IoU threshold (higher = better)
- **IoU** — how precisely the predicted polygon overlaps with the actual pothole boundary

---

## 💼 Business Impact

This project has direct applications in road maintenance, public safety, and smart city infrastructure:

- **Local councils & highways agencies** — automate road defect surveys, replacing expensive manual inspections
- **Smart city integration** — connect with dashcam networks or roadside CCTV for continuous 24/7 monitoring
- **Automotive ADAS** — embed in vehicles to alert drivers of upcoming road hazards in real time
- **Public transport operators** — protect fleet vehicles by detecting and routing around road damage
- **Insurance & risk** — provide objective, timestamped evidence of road conditions for liability assessments

> Polygon segmentation (used in this project) is more valuable than bounding-box detection for maintenance teams — it reveals the **exact shape and area** of each pothole, enabling severity classification and repair cost estimation.

---

## 📁 Project Structure

## Project Structure

```text
pothole-detection-yolov8/
├── results/
├── src/
│   └── inference.py
├── notebooks/
│   └── README.md
├── requirements.txt
├── LICENSE
└── README.md

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- pip
- A GPU is recommended for real-time inference

### 1. Clone the repository
```bash
git clone https://github.com/rishabrjk/pothole-detection-yolov8.git
cd pothole-detection-yolov8
```

### 2. Install dependencies
```bash
pip install ultralytics roboflow opencv-python numpy
```

### 3. Download the dataset from Roboflow
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("freedomtech").project("rpi4-yolov8-segmentation")
dataset = project.version(1).download("yolov8")
```

### 4. Train the model
```bash
yolo task=segment mode=train model=yolov8s-seg.pt data=datasets/data.yaml epochs=100 imgsz=640
```

### 5. Run real-time inference on a video
```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/best.pt")
class_names = model.names

cap = cv2.VideoCapture('your_road_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (1020, 500))
    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        masks = r.masks
        if masks is not None:
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (img.shape[1], img.shape[0]))
                contours, _ = cv2.findContours(
                    seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out.write(img)
    cv2.imshow('Pothole Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete!")
```

### 6. Test on a single image
```bash
yolo task=segment mode=predict model=models/best.pt source=data/images/test_road.jpg
```

---

## 📈 Sample Output

The model draws red polygon contours around each detected pothole, labelled with the class name `roadpothole` and confidence score. Multiple potholes in a single frame are individually segmented and labelled.

```
Detected: roadpothole — polygon contour drawn at [x:340, y:210]
Detected: roadpothole — polygon contour drawn at [x:120, y:300]
Detected: roadpothole — polygon contour drawn at [x:580, y:250]
Video processing complete!
```

---

## 🔮 Future Improvements

- [ ] Add pothole severity classification (minor / moderate / severe) based on contour area
- [ ] Integrate GPS tagging to map pothole locations automatically
- [ ] Deploy as a FastAPI REST endpoint for remote submission from dashcam devices
- [ ] Extend to multi-class detection: cracks, road markings, manholes
- [ ] Build a live monitoring dashboard showing detection frequency by road segment
- [ ] Incorporate ensemble methods and transfer learning from larger road defect datasets

---

## 👥 Team

**Rishab Kothari** (20261A0443)
**N. Rakesh Reddy** (20261A0437)
**G. Sathwik Reddy** (20261A0415)

Department of Electronics & Communication Engineering
Mahatma Gandhi Institute of Technology, Hyderabad — 2024

---

## 👤 Connect with Rishab

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/rishab-kothari-a09726220)
[![GitHub](https://img.shields.io/badge/GitHub-rishabrjk-black?style=flat-square&logo=github)](https://github.com/rishabrjk)
[![Email](https://img.shields.io/badge/Email-rishabkotharirj@gmail.com-red?style=flat-square&logo=gmail)](mailto:rishabkotharirj@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*If you found this project useful, please consider giving it a ⭐ — it helps others find it!*
