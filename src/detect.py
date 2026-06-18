from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model("test_images/test.jpg")

for r in results:
    print(r.boxes)