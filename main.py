from ultralytics import YOLO

if __name__ == "__main__":
    # โหลด Base model สำหรับ Classification
    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data="coffee_dataset",
        epochs=50,
        imgsz=224,
        batch=16,
        workers=2,
        device="cpu"
    )

    print("เทรนโมเดลเสร็จสมบูรณ์!")