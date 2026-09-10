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
# model = YOLO("yolov8n.pt")
# 1. Load trained YOLO object detection model
model = YOLO("runs/detect/train-5/weights/best.pt")

video_path = "conveyor_belt.mp4"
cap = cv2.VideoCapture(video_path)

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
        break

    # Run ByteTrack multi-object tracking
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml", verbose=False
    )

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        names = results[0].names

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Record trajectory and predicted label history
            track = track_history[track_id]
            track.append((cx, cy))
            class_history[track_id].append(names[cls_id])

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{track_id} {names[cls_id]}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
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