from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=r"C:\CoFFee\Coffee.v2i.folder",
    epochs=50,
    imgsz=224,
    batch=16,
    device=0
)

metrics = model.val(split="test")