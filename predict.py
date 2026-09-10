from ultralytics import YOLO

# 1. โหลดโมเดลตัวเก่งที่เทรนเสร็จแล้ว
model = YOLO("runs/classify/train-5/weights/best.pt")

# 2. ใส่ path ภาพเมล็ดกาแฟที่ต้องการทดสอบ (เช่น ภาพจากโฟลเดอร์ test)
img_path = r"C:\CoFFee\coffee_dataset\test\Fungus damage\Fungus-Damage_10_rotated_180_jpg.rf.e6cc5139e9c0984e6c47a5b8bd836039.jpg"

results = model.predict(source=img_path)

for r in results:
    top1_idx = r.probs.top1
    class_name = r.names[top1_idx]
    confidence = r.probs.top1conf.item() * 100
    
    print(f"Result: {class_name}")
    print(f"Confidence: {confidence:.2f}%")