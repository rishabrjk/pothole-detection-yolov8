# Real-Time Pothole Detection using YOLOv8

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=flat-square&logo=opencv)
![Roboflow](https://img.shields.io/badge/Roboflow-Annotated-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

Final year B.Tech major project developing a real-time pothole detection and segmentation system using YOLOv8, PyTorch, OpenCV, and Roboflow.

The model was trained on 2,023 annotated road images with polygon segmentation masks and used for video-based pothole detection.

---

## Project Overview

Road potholes contribute to vehicle damage, road maintenance costs, and safety risks, especially when inspections rely heavily on manual surveys.

This project builds an automated pothole detection and segmentation pipeline using YOLOv8 instance segmentation. Instead of only drawing bounding boxes, the system predicts polygon contours around potholes, providing more precise shape-level information for road defect detection.

---

## Problem Statement

Manual road inspection is time-consuming, inconsistent, and difficult to scale across large road networks.

This project aimed to build a computer vision system that can:

- Detect potholes from road images and video frames
- Segment potholes using polygon masks
- Support real-time inference using OpenCV
- Provide visual outputs useful for road maintenance and smart-city monitoring

---

## Dataset

The dataset was prepared using Roboflow.

Key details:

- 2,023 annotated road images
- Polygon segmentation masks
- Images collected from road scenes across varied lighting, surface, and environmental conditions
- Dataset exported in YOLOv8 segmentation format
- Image preprocessing included resizing to 640×640 and augmentation

---

## Tools Used

| Category | Tools |
|---|---|
| Programming | Python |
| Deep Learning | YOLOv8, PyTorch, Ultralytics |
| Computer Vision | OpenCV |
| Annotation | Roboflow |
| Data Handling | NumPy, Pandas |
| Visualisation | Matplotlib |
| Environment | Jupyter Notebook / Google Colab |
| Version Control | Git, GitHub |

---

## Methodology

1. Collected road images and video footage from public road-scene sources.
2. Annotated potholes using Roboflow polygon segmentation masks.
3. Applied preprocessing and augmentation, including resizing, flipping, brightness variation, and mosaic augmentation.
4. Fine-tuned YOLOv8s-seg on the annotated pothole dataset.
5. Evaluated the model using precision, recall, mAP, and IoU-based metrics.
6. Built an OpenCV inference pipeline to process video frames and draw pothole contours.

---

## Model Details

| Item | Description |
|---|---|
| Model | YOLOv8s-seg |
| Task | Instance segmentation |
| Dataset size | 2,023 annotated images |
| Annotation type | Polygon segmentation masks |
| Training epochs | 100 |
| Image size | 640×640 |
| Framework | Ultralytics YOLOv8 / PyTorch |
| Evaluation metrics | Precision, Recall, mAP@0.5, mAP@0.5:0.95, IoU |

---

## Results Summary

The model was able to detect potholes and generate segmentation contours around road defects in image and video inputs.

Training curves showed decreasing loss across 100 epochs, while precision, recall, and mAP improved steadily during training. The final system demonstrated a working real-time detection pipeline using OpenCV.

### Metric Meaning

- **Precision:** how many predicted potholes were correct
- **Recall:** how many actual potholes were detected
- **mAP@0.5:** detection quality at 50% IoU threshold
- **IoU:** overlap between predicted segmentation mask and actual pothole region

---

## Sample Outputs

The model draws polygon contours around detected potholes and labels each detection with class name and confidence score.

Add output screenshots or prediction examples inside the `results/` folder.

Example output:

```text
Detected: roadpothole — polygon contour drawn
Detected: roadpothole — confidence score displayed
Video processing complete
