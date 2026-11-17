import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=50, imgsz=640, batch=8, device=device, workers=2, patience=20,
    hsv_h=0.005, hsv_s=0.5, hsv_v=0.4,
    fliplr=0.5, flipud=0.5, degrees=180,
    name="fresas_3clases", exist_ok=True,
)
