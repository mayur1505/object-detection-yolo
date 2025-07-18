from ultralytics import YOLO
import cv2  
import cvzone
import math
from sort import *

#cap = cv2.VideoCapture(0) # For webcam use 0
# cap.set(3, 640)
# cap.set(4, 480)
cap = cv2.VideoCapture('Demo Videos/store-in.mp4')


model = YOLO('Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('Project 4 - Store Detector/mask.png')

#Tracking
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)


limits = [612,72, 826, 106]

totalCount = []

while True:
    success, img = cap.read()
    imgReigon = cv2.bitwise_and(img, mask)

    results = model(imgReigon, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1


            # Confidence
            conf = math.ceil((box.conf[0]*100))/100   # Confidence percentage
            #cvzone.putTextRect(img, f'{conf}', (max(0,x1), max(35,y1)))

            # Class name
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h),l=15,rt=5)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=1.5, thickness=1, offset=3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
            else:
                pass

    resultsTracker = tracker.update(detections)

    cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        print(result)
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h),l=15,rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=3, offset=10)
        
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 200, 0), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[3]+20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50),
                                   scale=2, thickness=3, offset=10)
    cv2.imshow("Image", img)
    #cv2.imshow("ImageReigon", imgReigon)
    cv2.waitKey(1)