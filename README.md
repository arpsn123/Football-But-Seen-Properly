<h1 align="center">⚽ SuperHuman Moments in Football, BUT SEEN PROPERLY... </h1>
<h2 align="center">Advanced Real-Time Object Detection & Tracking using YOLOv8 + ByteTrack + Supervision </h2>

<!-- Repository Overview Badges --> <div align="center"> <img src="https://img.shields.io/github/stars/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=ffca28"> <img src="https://img.shields.io/github/forks/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=00aaff"> <img src="https://img.shields.io/github/watchers/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=00e676"> </div> <!-- Issue & PR --> <div align="center"> <img src="https://img.shields.io/github/issues/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=ea4335"> <img src="https://img.shields.io/github/issues-pr/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=ff9100"> </div> <!-- Activity --> <div align="center"> <img src="https://img.shields.io/github/last-commit/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=673ab7"> <img src="https://img.shields.io/github/contributors/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=388e3c"> <img src="https://img.shields.io/github/repo-size/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=303f9f"> </div> <!-- Languages --> <div align="center"> <img src="https://img.shields.io/github/languages/count/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=607d8b"> <img src="https://img.shields.io/github/languages/top/arpsn123/Football-But-Seen-Properly?style=for-the-badge&logo=github&logoColor=white&color=4caf50"> </div> <!-- Status --> <div align="center"> <img src="https://img.shields.io/badge/Maintenance-Active-brightgreen?style=for-the-badge&logo=github&logoColor=white"> </div>


## 📌 Overview

This project presents a **high-performance computer vision pipeline** designed to detect, track, and analyze dynamic football scenes from video data. The system leverages **state-of-the-art deep learning (YOLOv8)** combined with **multi-object tracking (ByteTrack)** and **visual orchestration tools (Supervision)** to identify and follow multiple entities such as players, ball, and referees in real time.

The goal is to capture **"superhuman moments"** — fast-paced, high-intensity interactions — by building a robust perception system capable of operating under:

* Occlusions
* Rapid motion
* Scale variation
* Dense object environments

---

## 🧠 Core Concepts & Theoretical Foundations

### 🔷 1. Why Object Detection in Football?

Football is a **multi-agent dynamic system** where understanding spatial-temporal interactions is critical. Object detection enables:

* Player tracking
* Ball trajectory analysis
* Tactical pattern extraction
* Event detection (passes, shots, interceptions)


### 🔷 2. Why YOLOv8?

YOLO (You Only Look Once) is a **single-stage detector** that performs detection in one forward pass.

#### ✔ Advantages:

* Real-time performance (low latency)
* End-to-end training
* High mAP with efficient inference
* Anchor-free detection (in YOLOv8)

#### ✔ Why YOLOv8 specifically?

* Improved backbone (C2f modules)
* Decoupled head (better classification & localization)
* Better small object detection (important for ball detection)
* Native segmentation support



### 🔷 3. Why NOT Traditional Methods?

| Method           | Limitation                      |
| ---------------- | ------------------------------- |
| Haar Cascades    | Poor generalization             |
| HOG + SVM        | Slow, not scalable              |
| RCNN / Fast RCNN | Too slow for real-time          |
| Faster RCNN      | Still heavy for video pipelines |

👉 **Conclusion:** Only YOLO-style architectures meet **real-time + accuracy** constraints.



### 🔷 4. Why ByteTrack?

Object detection alone is insufficient — we need **identity persistence across frames**.

#### ✔ ByteTrack Key Idea:

Instead of discarding low-confidence detections, it:

* Associates **high-confidence detections first**
* Then uses **low-confidence detections** for recovery

#### ✔ Benefits:

* Better tracking in occlusions
* Reduced ID switching
* Higher tracking accuracy (MOTA)



### 🔷 5. Why Supervision Library?

Supervision acts as a **pipeline abstraction layer** for:

* Detection visualization
* Tracking integration
* Annotation rendering
* Video processing utilities

#### ✔ Why use it?

* Cleaner, modular code
* Faster development
* Built-in utilities for bounding boxes, labels, traces



### 🔷 6. Why Frame-by-Frame Processing?

Video = sequence of frames → processed individually.

#### ✔ Reasons:

* Deep learning models operate on images, not streams
* Enables temporal tracking via trackers
* Memory efficient
* Allows fine-grained control per frame



### 🔷 7. Why Generator-Based Video Processing?

Using generators instead of loading full video:

#### ✔ Benefits:

* Memory efficient (no full video load)
* Scalable for long videos
* Enables streaming pipelines

---

## ⚙️ System Architecture

```
Video Input
   ↓
Frame Generator
   ↓
YOLOv8 Detection
   ↓
ByteTrack Tracking
   ↓
Supervision Annotation
   ↓
Output Video
```

---

## 📦 Dependencies

* `ultralytics` → YOLOv8
* `supervision` → visualization & utilities
* `opencv-python` → video processing
* `numpy` → numerical operations

---

## 📥 Model Initialization

```python
from ultralytics import YOLO

model = YOLO("yolov8x.pt")  # Using largest variant for maximum accuracy
```

---

## 🎯 Class Mapping

```python
CLASS_NAMES_DICT = model.model.names
```

Typical COCO classes include:

* person
* sports ball
* referee (mapped via person class)
* etc.

---

## 🎥 Frame Generator

```python
import cv2

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        yield frame
    
    cap.release()
```

---

## 🔍 Detection Pipeline

```python
def detect(frame):
    results = model(frame)[0]
    return results
```

---

## 🧭 Tracking with ByteTrack

```python
import supervision as sv

tracker = sv.ByteTrack()

def track(detections):
    tracked = tracker.update_with_detections(detections)
    return tracked
```

---

## 🎨 Annotation Layer

```python
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def annotate(frame, detections):
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    
    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels=labels)
    
    return frame
```

---

## 🔄 Full Pipeline

```python
for frame in generate_frames("input.mp4"):
    
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    detections = tracker.update_with_detections(detections)
    
    annotated_frame = annotate(frame, detections)
    
    cv2.imshow("Output", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
```

---
<img width="1260" height="717" alt="e02bf30c-4136-4f0f-9298-acbf9b941c4f" src="https://github.com/user-attachments/assets/56f96e1b-3f45-47f6-8731-ebba45402e01" />

## 🚀 Performance Considerations

| Factor               | Impact                     |
| -------------------- | -------------------------- |
| Model size (YOLOv8x) | High accuracy, lower FPS   |
| Resolution           | Higher → slower inference  |
| GPU                  | Critical for real-time     |
| Batch size           | Typically 1 (video stream) |

---

## 📊 Key Challenges Addressed

* Motion blur
* Occlusion handling
* Multi-object identity preservation
* Small object detection (ball)
* Real-time constraints

---

## 🔬 Research Extensions

This pipeline can be extended into:

* Player re-identification (ReID)
* Tactical heatmaps
* Pass detection using trajectory modeling
* Action recognition (LSTM / Transformers)
* Ball possession analytics

---

## 🧠 Future Improvements

* Fine-tuned YOLO on football-specific dataset
* Integration with pose estimation (e.g., OpenPose)
* Event detection (goals, fouls)
* Multi-camera fusion

---

## 📌 Conclusion

This project demonstrates a **production-grade computer vision system** combining:

* **Detection (YOLOv8)**
* **Tracking (ByteTrack)**
* **Visualization (Supervision)**

The design choices emphasize:

* Real-time capability
* Robust tracking under uncertainty
* Modular and scalable architecture

---
## 🙏 Acknowledgements

The input video used in this project was sourced from **YouTube**. All credits for the original footage belong to the respective content creators and rights holders.
This project extensively utilizes the **Roboflow ecosystem**, particularly the **Supervision library**, which played a crucial role.

---

https://github.com/user-attachments/assets/10a4cf2c-702f-400f-85cc-0df16c151bf2



