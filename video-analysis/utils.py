import cv2
import numpy as np
import colorsys
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def load_model(model_name="yolov8n.pt"):
    """
    Load a YOLO detection model.

    Parameters
    ----------
    model_name : str, optional (default="yolov8n.pt")
        Path or name of the YOLO model to load. Can be a local file or a model from the Ultralytics hub.

    Returns
    -------
    YOLO
        An instance of the YOLO model ready for inference.

    """
    return YOLO(model_name)


def pad_frame(img, pad=16)-> np.ndarray:
    """
    Add a black padding around an image.

    Parameters
    ----------
    img : numpy.ndarray
        The input image to be padded.
    pad : int, optional (default=16)
        The number of pixels to add on each side of the image.

    Returns
    -------
    numpy.ndarray
        The padded image with a black border.
    """
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def unpad_frame(img, pad=16) -> np.ndarray:
    """
    Remove padding from an image.

    Parameters
    ----------
    img : numpy.ndarray
        The padded input image.
    pad : int, optional (default=16)
        The number of pixels as pad

    Returns
    -------
    numpy.ndarray
        The unpadded image.
    """
    if pad <= 0:
        return img
    return img[pad:-pad, pad:-pad]


def overlay_counts(frame, per_frame_count, unique_count) -> np.ndarray:
    """
    Overlay current and unique counts on a video frame in a compact style.

    Parameters
    ----------
    frame : np.ndarray
        The image on which to overlay the counts.
    per_frame_count : int
        Count of objects in the current frame.
    unique_count : int
        Total unique objects detected so far.

    Returns
    -------
    np.ndarray
        Frame with counts overlaid.
    """
    h, w, _ = frame.shape

    padding = 8
    rect_w, rect_h = 140, 50
    x1, y1 = padding, h - rect_h - padding
    x2, y2 = x1 + rect_w, y1 + rect_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    color = (255, 255, 255)

    cv2.putText(frame, f"Now: {per_frame_count}", (x1 + 10, y1 + 20), font, scale, color, thickness)
    #cv2.putText(frame, f"Unique: {unique_count}", (x1 + 10, y1 + 40), font, scale, color, thickness)

    return frame

def draw_transparent_box(frame, box, color=(0, 255, 0), alpha=0.1, thickness=2)-> np.ndarray:
    """
    Draw a semi-transparent rectangle on the frame.

    Parameters
    ----------
    frame : np.ndarray
        Original frame (BGR)
    box : tuple
        (x1, y1, x2, y2)
    color : tuple
        BGR color of the rectangle
    alpha : float
        Transparency factor (0=fully transparent, 1=opaque)
    thickness : int
        Thickness of rectangle border
    Returns
    -------
    frame : numpy.ndarray
        The frame with the counts overlaid.
    """
    overlay = frame.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


track_frames = {}  # track_id -> number of frames seen
track_history = {}  # track_id -> list of previous boxes
track_colors = {}  # track_id -> BGR color


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """
    Assign a different color to each track ID.

    Parameters
    ----------
    track_id : int
        The unique identifier for the track.

    Returns
    -------
    tuple[int, int, int]
        The RGB color assigned to the track, values in the range 0-255.
    """
    if track_id not in track_colors:
        hue = (int(track_id) * 37 % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        track_colors[track_id] = (
            int(b * 255),
            int(g * 255),
            int(r * 255),
        )
    return track_colors[track_id]

def shrink_box(box, shrink_factor=0.8):
    """
    Shrink a bounding box by a factor, keeping the center the same.

    Parameters
    ----------
    box : list or tuple
        [x1, y1, x2, y2]
    shrink_factor : float
        Factor to reduce width and height (0 < factor <= 1)

    Returns
    -------
    list
        Shrunk box coordinates [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * shrink_factor, (y2 - y1) * shrink_factor
    x1_new = cx - w / 2
    y1_new = cy - h / 2
    x2_new = cx + w / 2
    y2_new = cy + h / 2
    return [x1_new, y1_new, x2_new, y2_new]


def process_frame(
    frame,
    model,
    tracker_type: str = "deepsort",
    tracker=None,
    unique_ids: set = None,
    conf: float = 0.50,
    iou: float = 0.50,
    padding: int = 16,
) -> tuple:
    """
    Process a video frame for object detection and tracking.

    This function supports 'deepsort' and 'bytetrack' tracking methods. It
    detects objects in the frame, updates the tracker, annotates the frame
    with bounding boxes, IDs, and counts, and returns the annotated frame
    and the set of unique track IDs.

    Parameters
    ----------
    frame : np.ndarray
        The input video frame (H x W x 3).
    model : object
        The detection/tracking model (YOLO, etc.).
    tracker_type : str, optional
        Tracker type to use: 'deepsort' or 'bytetrack'. Default is 'deepsort'.
    tracker : object, optional
        Tracker object (required if tracker_type is 'deepsort').
    unique_ids : set, optional
        Set to store unique track IDs across frames. If None, a new set is created.
    conf : float, optional
        Confidence threshold for detections. Default is 0.20.
    iou : float, optional
        Intersection-over-union threshold for detections. Default is 0.50.
    padding : int, optional
        Number of pixels to pad around the frame

    Returns
    -------
    tuple
        annotated : np.ndarray
            The frame annotated with bounding boxes, track IDs, and counts.
        unique_ids : set
            Set of unique track IDs detected so far.
    """
    frame_in = pad_frame(frame, pad=padding) if padding else frame
    per_frame_count = 0
    if unique_ids is None:
        unique_ids = set()

    if tracker_type.lower() == "bytetrack":
        results = model.track(
            source=frame_in,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=conf,
            iou=iou,
            persist=True,
            verbose=False,
        )
        r = results[0]
        annotated = r.plot()
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            unique_ids.update(ids)
        per_frame_count = 0 if r.boxes is None else len(r.boxes)

    elif tracker_type.lower() == "deepsort":
        results = model(frame_in, conf=conf, iou=iou, classes=[0])[0]
        boxes = results.boxes.xyxy.cpu().numpy()  # Nx4
        scores = results.boxes.conf.cpu().numpy()  # N

        detections = [
             (shrink_box(box.tolist()), float(score), 0) for box, score in zip(boxes, scores)
        ]

        tracks = tracker.update_tracks(detections, frame=frame)


        annotated = frame_in.copy()
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            track_frames[track_id] = track_frames.get(track_id, 0) + 1
            if track_frames[track_id] < 3:
                continue
            unique_ids.add(track_id)
            #x1, y1, x2, y2 = map(int, track.to_ltrb())
            x1, y1, x2, y2 = smooth_track(track_id, track.to_ltrb())
            annotated = draw_transparent_box(
                annotated,
                (x1, y1, x2, y2),
                color=get_track_color(track_id),
                alpha=0.05,
                thickness=0,
            )
            cv2.putText(
                annotated,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                get_track_color(track_id),
                2,
            )
            per_frame_count += 1

    annotated = unpad_frame(annotated, pad=padding)
    annotated = overlay_counts(annotated, per_frame_count, len(unique_ids))

    return annotated, unique_ids



def smooth_track(track_id: int, box: tuple, history_len: int = 5) -> tuple:
    """
    Smooth the bounding box of a tracked person.

    Parameters
    ----------
    track_id : int
        The unique ID of a person.
    box : tuple
        The current bounding box (x1, y1, x2, y2) of the object.
    history_len : int, optional
        Number of previous boxes to consider for smoothing.

    Returns
    -------
    tuple
        Smoothed bounding box (x1, y1, x2, y2).
    """
    if track_id not in track_history:
        track_history[track_id] = []
    track_history[track_id].append(box)

    if len(track_history[track_id]) > history_len:
        track_history[track_id].pop(0)

    # Average coordinates
    x1 = int(sum(b[0] for b in track_history[track_id]) / len(track_history[track_id]))
    y1 = int(sum(b[1] for b in track_history[track_id]) / len(track_history[track_id]))
    x2 = int(sum(b[2] for b in track_history[track_id]) / len(track_history[track_id]))
    y2 = int(sum(b[3] for b in track_history[track_id]) / len(track_history[track_id]))

    # Clamp box size relative to previous
    if len(track_history[track_id]) > 1:
        prev_box = track_history[track_id][-2]
        max_width = int((prev_box[2] - prev_box[0]) * 2)
        max_height = int((prev_box[3] - prev_box[1]) * 2)
        width = min(x2 - x1, max_width)
        height = min(y2 - y1, max_height)
        x2 = x1 + width
        y2 = y1 + height

    return (x1, y1, x2, y2)



def process_video(
    video_path: str,
    output_path: str,
    model,
    tracker_type: str = "bytetrack",
    conf: float = 0.5,
    iou: float = 0.5
) -> set:
    """
    Process a video to detect and track objects, saving an annotated output video.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_path : str
        Path to save the processed output video.
    model : object
        Object detection model (e.g., YOLO) used for detecting objects in frames.
    tracker_type : str, optional
        Type of tracker to use ("deepsort" or "bytetrack"). Default is "deepsort".
    conf : float, optional
        Confidence threshold for detections. Default is 0.2.
    iou : float, optional
        Intersection-over-Union threshold for detections. Default is 0.5.

    Returns
    -------
    set
        Set of unique track IDs encountered during processing.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracker
    tracker = None
    if tracker_type.lower() == "deepsort":
        tracker = DeepSort(max_age=45, n_init=5, max_cosine_distance=0.2)
    unique_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, unique_ids = process_frame(
            frame, model, tracker_type, tracker, unique_ids, conf=conf, iou=iou
        )
        out.write(annotated)

    cap.release()
    out.release()
    return unique_ids
