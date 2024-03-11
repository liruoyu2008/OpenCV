'''Shi-Tomasi角点检测'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./images/9.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 找到8个最佳角点
corners = cv.goodFeaturesToTrack(gray,8,0.01,5)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.show()