mport os
from ultralytics import YOLO

model = YOLO("scl.pt")

results = model.train(data="mydataset.yaml", epochs=100)
