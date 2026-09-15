from collections import Counter, defaultdict
import cv2
from ultralytics import YOLO


def ccw(A, B, C):
    """Check if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def is_intersecting(p1, p2, line_start, line_end):
    """Check if line segment p1-p2 intersects line segment line_start-line_end."""
    return (ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end)) and (
        ccw(p1, p2, line_start) != ccw(p1, p2, line_end)
    )


# ================= Models Initialization =================
# Stage 1: Detector model for localization and tracking
detector =  YOLO("runs/detect/runs/detect/bean_detector-6/weights/best.pt")

# Stage 2: Classifier model (16 classes) for defect identification
classifier = YOLO("backend/runs/classify/train-5/weights/best.pt")

# video_path = r"C:\CoFFee\conveyor_belt.mp4"
# cap = cv2.VideoCapture(video_path)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# Virtual counting line coordinates (adjust to match camera viewpoint)
LINE_START = (50, 400)
LINE_END = (600, 400)

# Tracking and counting state containers
track_history = defaultdict(lambda: [])
class_history = defaultdict(lambda: [])
counted_ids = set()
final_counts = Counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    h_img, w_img, _ = frame.shape

    # Stage 1: Run ByteTrack multi-object tracking
    results = detector.track(
        frame, persist=True, tracker="bytetrack.yaml", verbose=False
    )

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Keep crop coordinates within valid frame boundaries
            crop_x1 = max(0, x1)
            crop_y1 = max(0, y1)
            crop_x2 = min(w_img, x2)
            crop_y2 = min(h_img, y2)

            # Stage 2: Crop bean region and classify defect
            cropped_bean = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped_bean.size > 0:
                cls_res = classifier(cropped_bean, verbose=False)
                top1_idx = cls_res[0].probs.top1
                pred_class = cls_res[0].names[top1_idx]
            else:
                pred_class = "Unknown"

            # Record trajectory and predicted label history
            track = track_history[track_id]
            track.append((cx, cy))
            class_history[track_id].append(pred_class)

            if len(track) > 30:
                track.pop(0)

            # Check line intersection on tracked objects
            if len(track) >= 2 and track_id not in counted_ids:
                prev_point = track[-2]
                curr_point = track[-1]

                if is_intersecting(
                    prev_point, curr_point, LINE_START, LINE_END
                ):
                    # Use majority voting across observed frames for class stability
                    votes = Counter(class_history[track_id])
                    best_class = votes.most_common(1)[0][0]

                    final_counts[best_class] += 1
                    counted_ids.add(track_id)

            # Draw bounding box and identifier
            color = (0, 255, 0) if pred_class.lower() == "good" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID:{track_id} {pred_class}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    # Render line and on-screen metrics overlay
    cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 2)
    y_offset = 30
    for cls_name, cnt in final_counts.items():
        cv2.putText(
            frame,
            f"{cls_name}: {cnt}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y_offset += 25

    cv2.imshow("Conveyor Inspection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ================= Final Inspection Report =================
total_beans = sum(final_counts.values())

print("\n" + "=" * 48)
print("       CONVEYOR SORTING SUMMARY REPORT       ")
print("=" * 48)
print(f"Total Beans Inspected: {total_beans}\n")

if total_beans > 0:
    for cls_name, count in final_counts.items():
        percent = (count / total_beans) * 100
        print(f"- {cls_name:<20}: {count:>4} ({percent:>6.2f}%)")
print("=" * 48)
