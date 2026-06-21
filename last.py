from ultralytics import YOLO

model = YOLO("runs/detect/runs/fruit_v1/weights/last.pt")
model.train(resume=True)
