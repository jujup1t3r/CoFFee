from ultralytics import YOLO

def main():
    # โหลดโมเดลตั้งต้นสำหรับ Object Detection
    model = YOLO("yolov8n.pt")

    # เทรนโมเดลหาตำแหน่งเมล็ดกาแฟ
    model.train(
        data="Coffee Bean.v5/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        single_cls=True,
        cache='disk',
        project="runs/detect",
        name="bean_detector",
        device=0  # บังคับใช้ GPU ตัวที่ 0 (RTX 3050)
    )

if __name__ == '__main__':
    main()