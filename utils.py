from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

SMOOTH_FRAMES = 5  # number of frames to smooth counts over

def load_model(model_name="yolov8x.pt"):
    return YOLO(model_name)

def pad_frame(img, pad=16):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))

def smooth_count(count, history=[]):
    history.append(count)
    if len(history) > SMOOTH_FRAMES:
        history.pop(0)
    return int(np.mean(history))

def draw_transparent_boxes(frame, boxes, ids=None, alpha=0.4):
    overlay = frame.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = (37 * int(ids[i]) % 255, 17 * int(ids[i]) % 255, 29 * int(ids[i]) % 255) if ids else (0,255,0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        if ids:
            cv2.putText(overlay, f"ID {ids[i]}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def draw_overlay(frame, per_frame_count, unique_count=None):
    h, w, _ = frame.shape
    overlay = frame.copy()
    font, font_scale, thickness, padding = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 10

    text_now = f"Now: {per_frame_count}"
    text_unique = f"Unique: {unique_count}" if unique_count is not None else ""
    (tw1, th1), _ = cv2.getTextSize(text_now, font, font_scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(text_unique, font, font_scale, thickness)
    rect_width = max(tw1, tw2) + 2*padding
    rect_height = th1 + (th2 if text_unique else 0) + 3*padding

    x1, y1 = 10, h - rect_height - 10
    x2, y2 = x1 + rect_width, y1 + rect_height
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, text_now, (x1+padding, y1+padding+th1), font, font_scale, (255,255,255), thickness)
    if text_unique:
        cv2.putText(frame, text_unique, (x1+padding, y1+2*padding+th1+th2//2), font, font_scale, (255,255,255), thickness)
    return frame

def process_frame(frame, model, tracker, conf=0.3, iou=0.5, padding=16, count_history=[]):
    # Pad frame for YOLO
    frame_in = pad_frame(frame, pad=padding) if padding else frame

    # Run YOLO detection
    results = model(frame_in, conf=conf, iou=iou, classes=[0])[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # Nx4
    scores = results.boxes.conf.cpu().numpy()  # N

    # Unpad boxes if needed
    if padding > 0:
        boxes[:, [0, 2]] -= padding
        boxes[:, [1, 3]] -= padding

    # Prepare detections for DeepSORT
    detections = [(box.tolist(), float(score), 0) for box, score in zip(boxes, scores)]
    tracks = tracker.update_tracks(detections, frame=frame)

    # Keep only confirmed tracks
    confirmed_tracks = [t for t in tracks if t.is_confirmed()]
    ids = [int(t.track_id) for t in confirmed_tracks]

    # Use YOLO boxes for display
    track_boxes = boxes.tolist()  # keep original YOLO boxes
    per_frame_count = smooth_count(len(ids), count_history)

    # Annotate frame with semi-transparent boxes + IDs
    annotated = draw_transparent_boxes(frame, track_boxes, ids)

    return annotated, ids, per_frame_count


def process_video(video_in, video_out, model, conf=0.3, iou=0.5, padding=16):
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = DeepSort(max_age=90, n_init=3, max_iou_distance=0.7)
    unique_ids = set()
    count_history = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, ids, per_frame_count = process_frame(frame, model, tracker, conf, iou, padding, count_history)
        unique_ids.update(ids)
        annotated = draw_overlay(annotated, per_frame_count, len(unique_ids))
        out.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    return unique_ids
