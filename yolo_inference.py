from ultralytics import YOLO

# model = YOLO('yolov8m')
model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True)

print(results[0])
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
for box in results[0].boxes:
    print(box)
    