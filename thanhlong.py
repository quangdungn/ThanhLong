mport os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="mydataset.yaml", epochs=100)