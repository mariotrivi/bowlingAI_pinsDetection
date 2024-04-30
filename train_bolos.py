from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data="/workspaces/ultralytics/datasets/bowling/data.yaml", epochs=200, imgsz=640, batch=8, workers=0)