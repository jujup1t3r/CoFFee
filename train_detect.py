from ultralytics import YOLO

def main():
    # โหลดโมเดลตั้งต้นสำหรับ Object Detection
    model = YOLO("yolov8n.pt")

    # เทรนโมเดลหาตำแหน่งเมล็ดกาแฟ
    model.train(
        data="Coffee Bean.v5/data.yaml",
        epochs=15,
        imgsz=320,
        batch=16,
        single_cls=True,
        cache=True,
        project="runs/detect",
        name="bean_detector",
        device=0  # บังคับใช้ GPU ตัวที่ 0 (RTX 3050)
    )

if __name__ == '__main__':
    main()