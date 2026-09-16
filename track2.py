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

# ==================== VIDEO / CAMERA SETUP ====================
# สลับใช้กล้องจริง: cap = cv2.VideoCapture(0)
VIDEO_PATH = "test3.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[Error] Cannot open video source: {VIDEO_PATH}")
    exit()

def is_skin(crop_img):
    """ตรวจจับเฉดสีผิวคนเพื่อตัดนิ้วมือออก"""
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = cv2.countNonZero(mask) / (crop_img.shape[0] * crop_img.shape[1])
    return skin_ratio > 0.65

track_predictions = defaultdict(list)
counted_ids = set()
final_counts = defaultdict(int)

# ==================== MAIN INSPECTION LOOP ====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = detector.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.45,
        iou=0.65,
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

            # 1. กรองขนาดเมล็ดกาแฟจริง
            if not (15 <= w <= 160 and 15 <= h <= 160 and 300 <= area <= 22000):
                continue

            # 2. กรองสัดส่วน
            if not (0.5 <= aspect_ratio <= 1.9):
                continue

            # ตัดรูปพร้อม padding
            pad = 6
            h_f, w_f = frame.shape[:2]
            crop = frame[max(0, y1 - pad):min(h_f, y2 + pad), max(0, x1 - pad):min(w_f, x2 + pad)]
            if crop.size == 0 or is_skin(crop):
                continue

            # 3. จำแนกประเภท
            cls_res = classifier.predict(crop, verbose=False)[0]
            top1_conf = float(cls_res.probs.top1conf.item())
            top1_cls = cls_res.names[cls_res.probs.top1]

            if top1_conf >= 0.60:
                track_predictions[track_id].append(top1_cls)

            # 4. ลอจิกนับทันทีเมื่อตรวจจับนิ่งพอ (อยู่นิ่งหรือขยับก็ตรวจได้ โดยดูจากประวัติ 5 เฟรม)
            if track_id not in counted_ids and len(track_predictions[track_id]) >= 5:
                counted_ids.add(track_id)
                
                # โหวตหาคลาสหลัก
                final_class = Counter(track_predictions[track_id]).most_common(1)[0][0]
                final_counts[final_class] += 1

                payload_event = {
                    "track_id": int(track_id),
                    "defect_type": str(final_class),
                    "timestamp": time.time(),
                    "is_defect": str(final_class).lower() != "good"
                }
                mqtt_client.publish(TOPIC_EVENT, json.dumps(payload_event))
                print(f"[MQTT] Confirmed Bean #{track_id} -> {final_class}")

            # กำหนดสีกรอบ: เมล็ดที่ยืนยันแล้ว vs กำลังวิเคราะห์
            cur_label = track_predictions[track_id][-1] if track_predictions[track_id] else "detecting..."
            if track_id in counted_ids:
                box_color = (0, 255, 0) if cur_label.lower() == "good" else (0, 0, 255)
                status_txt = f"ID:{track_id} {cur_label} (Counted)"
            else:
                box_color = (255, 255, 0)
                status_txt = f"ID:{track_id} Analyzing..."

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, status_txt, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # สรุปยอดรวมบนหน้าจอ
    y_offset = 30
    for cls_name, count in final_counts.items():
        cv2.putText(frame, f"{cls_name}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 25

    cv2.imshow("Stationary & Moving Bean Inspection", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # กดปุ่ม 'r' บนคีย์บอร์ดเพื่อรีเซ็ตในระหว่างรัน
        counted_ids.clear()
        track_predictions.clear()
        final_counts.clear()
        print("[System] Detection counts & Track IDs cleared!")

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