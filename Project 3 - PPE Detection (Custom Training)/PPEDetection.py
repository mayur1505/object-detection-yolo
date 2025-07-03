from ultralytics import YOLO
import cv2  
import cvzone
import math

#cap = cv2.VideoCapture(0)  # Use 0 for the default camera
# cap.set(3, 1280)  # Set width
# cap.set(4, 720)   # Set height
# cap.set(3, 640)  # Set width
# cap.set(4, 480)   # Set height
cap = cv2.VideoCapture('Demo Videos/ppe-1-1.mp4')  # Use a video file

#this model(best.pt) is trained on custom dataset for PPE detection
# you can find this traning in google colab notebook named Yolov8.ipynb
model = YOLO('Yolo-Weights/best.pt')

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 
              'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 
              'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 
              'fire hydrant', 'machinery', 'mini-van', 'sedan', 
              'semi', 'trailer', 'truck and trailer', 'truck', 
              'van', 'vehicle', 'wheel loader']

myColor=(0,0,255)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor,3)
            #Confidence
            conf = math.ceil((box.conf[0]*100))/100   # Confidence percentage
            #class name
            cls = int(box.cls[0])  # Class index
            currentClass = classNames[cls]
            if conf>0.5:
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0,255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                    myColor =(0,255,0)
                else:
                    myColor = (255, 0, 0)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                                    (max(0,x1),max(35,y1)),scale=1.5,thickness=1,colorB=myColor,
                                    colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor,3)


            

    cv2.imshow("Image", img)
    cv2.waitKey(1)