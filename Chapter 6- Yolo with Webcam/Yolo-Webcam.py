from ultralytics import YOLO
import cv2  
import cvzone
import math

#cap = cv2.VideoCapture(0)  # Use 0 for the default camera
# cap.set(3, 1280)  # Set width
# cap.set(4, 720)   # Set height
# cap.set(3, 640)  # Set width
# cap.set(4, 480)   # Set height
cap = cv2.VideoCapture('Demo Videos/store.mp4')  # Use a video file


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
 

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            #Confidence
            conf = math.ceil((box.conf[0]*100))/100   # Confidence percentage
            #class name
            cls = int(box.cls[0])  # Class index
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1),max(35,y1)),scale=1.5,thickness=1)

            

    cv2.imshow("Image", img)
    cv2.waitKey(1)