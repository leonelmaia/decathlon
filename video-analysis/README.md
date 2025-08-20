# Video Analytics – Customer Detection & Tracking
##  Project Overview

This project focuses on detecting and tracking customers visible in short store footage. The goal is to:

Detect customers frame by frame using an object detection model.

By combining detection with tracking, the pipeline allows for both per-frame counts (customers currently visible) and cumulative counts (unique customers seen so far).

## ⚙️ Tools & Methods
###  Detection

Model used: YOLOv11 (Ultralytics)

Reasoning: YOLO is state-of-the-art in real-time object detection, fast and accurate for human detection in store environments.

### Config:

--conf 0.5 → filters out low-confidence detections.

--iou 0.5 → sets the threshold for Non-Maximum Suppression (avoiding duplicate overlapping boxes).

##  Tracking

Two trackers were tested to compare performance:

### 1. ByteTrack

Strengths: Lightweight, produces tight bounding boxes around people.

Weaknesses: Can lose track IDs (ID switching), especially if customers overlap or occlude each other.

### 2. DeepSORT

Strengths: Better at maintaining consistent IDs across frames using both motion and appearance features.

Weaknesses: Bounding boxes can be larger and less precise compared to ByteTrack.

## Post-processing

Bounding box smoothing → reduces jitter by averaging positions over a few frames.

Box shrinkage (optional for DeepSORT) → tightens oversized boxes to resemble ByteTrack’s output.

## Running the Code
### Environment

Tests were conducted on Google Colab with a Tesla T4 GPU.

The GPU acceleration allowed faster inference, making real-time or near-real-time analysis possible even on longer videos.

Using Colab also simplified experimentation with different trackers (ByteTrack, DeepSORT) and YOLO configurations.

Example command:

``python video_analysis.py --input video.mp4 --output out.mp4 --model yolo11x.pt --conf 0.1 --iou 0.5``

##  Observations & Challenges

ByteTrack performed better in producing tight, visually appealing boxes, but struggled with ID consistency when customers moved close together.

DeepSORT maintained IDs more consistently, but its boxes tended to be oversized, sometimes including background.

Balancing IoU thresholds and tracker parameters was critical:

Higher IoU (e.g., 0.8) → stricter matching, more ID switches.

Lower IoU (e.g., 0.4–0.5) → smoother tracking, but risk of false matches.

Store environments with occlusion (e.g., customers walking close together) made tracking more difficult.

## Conclusion

YOLOv11 + ByteTrack → best for tight detection boxes.

YOLOv11 + DeepSORT → best for stable ID assignment.
