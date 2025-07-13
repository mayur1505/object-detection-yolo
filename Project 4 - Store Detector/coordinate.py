import cv2

# Load your video
cap = cv2.VideoCapture('Demo Videos/store-in.mp4')

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", click_event)

while True:
    success, frame = cap.read()
    if not success:
        break

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
