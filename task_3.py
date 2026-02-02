# Using the video track and trace the green ball (using opencv)
import cv2
import numpy as np
from collections import deque
cap = cv2.VideoCapture("data/Ball_Tracking.mp4")
if not cap.isOpened(): #check
    print("Error")
    exit()
#green clr identification
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])
# store
points = deque(maxlen=500)
while True: #framewise
    ret, frame = cap.read()
    if not ret: #end
        break
    frame = cv2.resize(frame, (640, 480)) #resize
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #hsv for more accurate green
    mask = cv2.inRange(hsv, lower_green, upper_green) #whie and black
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2) #noise removed
    contours, _ = cv2.findContours( 
        mask.copy(), #store
        cv2.RETR_EXTERNAL, #only outer
        cv2.CHAIN_APPROX_SIMPLE #only essetials
    )
    center = None
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea) #assumed as ball
        ((x, y), radius) = cv2.minEnclosingCircle(c) #circle around
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"]) 
            )#centre of mass check
        if radius > 5: #check 
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2) #green boundary circle
            cv2.circle(frame, center, 5, (0, 0, 255), -1) #red centre dot
    points.appendleft(center)
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2) #blue line trace
    cv2.imshow("Green Ball Tracking", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
