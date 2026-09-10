import cv2
from ultralytics import YOLO

model = YOLO("runs/classify/train-5/weights/best.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ส่งเฟรมเข้าโมเดล
    results = model(frame, verbose=False)
    
    # ดึงคลาสและค่าความมั่นใจ
    top1_idx = results[0].probs.top1
    label = results[0].names[top1_idx]
    conf = results[0].probs.top1conf.item() * 100

    # แสดงผลบนหน้าต่างวิดีโอ
    text = f"{label} ({conf:.1f}%)"
    color = (0, 0, 255) if "damage" in label.lower() or "fungus" in label.lower() else (0, 255, 0)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Coffee Bean Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()