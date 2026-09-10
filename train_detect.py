from ultralytics import YOLO

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
)