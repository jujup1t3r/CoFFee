import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
import paho.mqtt.client as mqtt

# ==================== MQTT CONFIGURATION ====================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_EVENT = "coffee/conveyor/bean_detected"
TOPIC_SUMMARY = "coffee/conveyor/summary"

try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[MQTT] Connected to broker successfully.")
    else:
        print(f"[MQTT] Connection failed with code {rc}")

mqtt_client.on_connect = on_connect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"[MQTT] Warning: Could not connect to MQTT broker ({e})")

# ==================== LOAD MODELS ====================
detector = YOLO("runs/detect/runs/detect/bean_detector-6/weights/best.pt")
classifier = YOLO("runs/classify/train-2/weights/best.pt")

# ==================== VIDEO & LINE SETUP ====================
# ใส่ชื่อไฟล์คลิป หรือ Full Path ของวิดีโอที่ต้องการทดสอบ
VIDEO_PATH = "test_conveyor.mp4" 

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[Error] Cannot open video file: {VIDEO_PATH}")
    exit()

LINE_START = (50, 350)
LINE_END = (600, 350)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def is_intersecting(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def is_skin(crop_img):
    """ตรวจจับเฉดสีผิวคนเพื่อตัดนิ้วมือออก"""
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = cv2.countNonZero(mask) / (crop_img.shape[0] * crop_img.shape[1])
    return skin_ratio > 0.65

track_history = defaultdict(list)
track_predictions = defaultdict(list)
counted_ids = set()
final_counts = defaultdict(int)

# ==================== MAIN INSPECTION LOOP ====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # เมื่อคลิปเล่นจนจบ ให้วนกลับไปเริ่มเฟรมแรกใหม่
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = detector.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.68,
        iou=0.4,
        verbose=False
    )

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect_ratio = float(w) / max(1, h)
            center_pt = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # 1. กรองขนาดเมล็ดกาแฟ
            if not (25 <= w <= 140 and 25 <= h <= 140 and 600 <= area <= 18000):
                continue

            # 2. กรองสัดส่วน
            if not (0.5 <= aspect_ratio <= 1.9):
                continue

            # ตัดรูปเมล็ดกาแฟ (พร้อมเพิ่ม padding 6 พิกเซล)
            pad = 6
            h_f, w_f = frame.shape[:2]
            crop = frame[max(0, y1 - pad):min(h_f, y2 + pad), max(0, x1 - pad):min(w_f, x2 + pad)]
            if crop.size == 0:
                continue

            # 3. ตัดนิ้วมือ
            if is_skin(crop):
                continue

            # 4. Classify เมล็ด
            cls_res = classifier.predict(crop, verbose=False)[0]
            top1_conf = float(cls_res.probs.top1conf.item())
            top1_cls = cls_res.names[cls_res.probs.top1]

            if top1_conf >= 0.60:
                track_predictions[track_id].append(top1_cls)

            # ตรวจสอบการตัดผ่านเส้นนับ
            prev_points = track_history[track_id]
            if len(prev_points) > 0:
                prev_pt = prev_points[-1]

                if is_intersecting(prev_pt, center_pt, LINE_START, LINE_END):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        classes_history = track_predictions[track_id]
                        final_class = Counter(classes_history).most_common(1)[0][0] if classes_history else "good"
                        final_counts[final_class] += 1

                        payload_event = {
                            "track_id": int(track_id),
                            "defect_type": str(final_class),
                            "timestamp": time.time(),
                            "is_defect": str(final_class).lower() != "good"
                        }
                        mqtt_client.publish(TOPIC_EVENT, json.dumps(payload_event))
                        print(f"[MQTT] Event Sent: Bean #{track_id} -> {final_class}")

            track_history[track_id].append(center_pt)
            if len(track_history[track_id]) > 25:
                track_history[track_id].pop(0)

            cur_label = track_predictions[track_id][-1] if track_predictions[track_id] else "detecting..."
            box_color = (0, 255, 0) if cur_label.lower() == "good" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"ID:{track_id} {cur_label}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # วาดเส้นตรวจสอบ
    cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 2)
    cv2.putText(frame, "Inspection Line", (LINE_START[0], LINE_START[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # สรุปผลบนหน้าต่าง
    y_offset = 30
    for cls_name, count in final_counts.items():
        cv2.putText(frame, f"{cls_name}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += 25

    cv2.imshow("Conveyor Inspection", frame)
    
    # หน่วงเวลา 30ms เพื่อให้ความเร็วเท่ากับคลิปจริง (~30 FPS)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# ==================== CLEANUP & SUMMARY ====================
summary_payload = {
    "total_inspected": sum(final_counts.values()),
    "counts": dict(final_counts),
    "timestamp": time.time()
}
mqtt_client.publish(TOPIC_SUMMARY, json.dumps(summary_payload), retain=True)
print("[MQTT] Final summary sent.")

mqtt_client.loop_stop()
mqtt_client.disconnect()
cap.release()
cv2.destroyAllWindows()