from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

# Load video and model
cap = cv2.VideoCapture('Demo Videos/store.mp4')
model = YOLO('Yolo-Weights/yolov8n.pt')

# Load class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load region mask
mask = cv2.imread('Project 4 - Store Detector/mask.png')

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Line for entry/exit detection (adjust as needed)
line = [(612, 115), (826, 115)]
line_y = line[0][1]

# Counters and tracking memory
total_entries = 0
total_exits = 0
track_history = {}



while True:
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    # Extract person detections from YOLO
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Track objects with SORT
    resultsTracker = tracker.update(detections)

    # Draw the counting line
    cv2.line(img, line[0], line[1], (0, 0, 255), 3)

    # Process tracked results
    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{id}', (x1, max(35, y1)), scale=2, thickness=3, offset=10)
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)

        # Track centroid history for direction check
        if id not in track_history:
            track_history[id] = [cy, cy]
        else:
            track_history[id][0] = track_history[id][1]
            track_history[id][1] = cy

            prev_y, curr_y = track_history[id]

            # Entry: top → bottom
            if prev_y < line_y and curr_y >= line_y:
                total_entries += 1
                cv2.line(img,(line[0]), (line[1]), (0, 255, 0), 5)
                print(f"Entry Detected (ID: {id})")
                del track_history[id]

            # Exit: bottom → top
            elif prev_y > line_y and curr_y <= line_y:
                total_exits += 1
                cv2.line(img,(line[0]), (line[1]), (0, 255, 0), 5)
                print(f"Exit Detected (ID: {id})")
                del track_history[id]

    # Display counts
    cvzone.putTextRect(img, f'Entries: {total_entries}', (20, 50), scale=2, thickness=2, offset=6)
    cvzone.putTextRect(img, f'Exits: {total_exits}', (20, 100), scale=2, thickness=2, offset=6)

    # Show final image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
