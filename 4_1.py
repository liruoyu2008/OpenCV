'''颜色空间'''

from types import MethodType
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
while (1):
    # 读取帧
    _, frame = cap.read()
    # 转换颜色空间 BGR 到 HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 定义HSV中黄色的范围
    # 色环0-180对应红、黄、绿、青、蓝、品，色调黄在15-45左右，饱和度要求较高（颜色纯度），亮度无要求
    lower_blue = np.array([15,180,0])
    upper_blue = np.array([45,255,255])
    # 设置HSV的阈值使得只取黄色
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # 将掩膜和图像逐像素相加
    res = np.zeros(frame.shape)
    res = cv.bitwise_and(frame, frame, res, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
