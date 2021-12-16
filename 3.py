import numpy as np
import cv2 as cv

cap = cv.VideoCapture('/Users/ryu/Movies/2021-12-17 00.11拍摄的影片.mov')
while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(25) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
