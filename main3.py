import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # or "yolov8s.pt" for faster/more accurate

# Load video
cap = cv2.VideoCapture("/home/shaswat22/telegram/1900-151662242_medium.mp4")

# COCO vehicle classes mapping
vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Dictionary to count vehicles per frame
    vehicle_count = {"Car":0, "Motorcycle":0, "Bus":0, "Truck":0}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, class_ids):
            cls = int(cls)
            if cls in vehicle_classes:
                vehicle_type = vehicle_classes[cls]
                vehicle_count[vehicle_type] += 1

                x1, y1, x2, y2 = map(int, box)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                # Put label with vehicle type and current count
                cv2.putText(frame, f"{vehicle_type} {vehicle_count[vehicle_type]}",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show total counts at top-left corner
    y0 = 30
    for v_type, count in vehicle_count.items():
        cv2.putText(frame, f"{v_type}: {count}", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        y0 += 30

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()