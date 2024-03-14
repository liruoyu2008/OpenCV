'''SURF加速稳健特征
主要是利用积分图、Hessian矩阵、盒子滤波器（box filter）全面替代了DoG算法从而加速了SIFT过程
目前高版本的SURF算法有专利保护无法使用，可以回退到opencv-python-2.4.2以使用cv.xfeatures2d.SURF_create'''
import numpy as np
import cv2 as cv

img = cv.imread('./images/15.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 创建surf对象并指定参数，还可以指定参数
# 这里可以指定hessian阈值为400（通常为300-500）
# 但400时得到的特征点太多，无法在图像上很好的呈现，因此这里刻意将其设置为50000
surf = cv.xfeatures2d.SURF_create(400)
# kp是关键点信息，des为关键点KPD
kp, des = surf.detectAndCompute(img, None)
img = cv.drawKeypoints(gray, kp, None, (255, 0, 0), 4)
cv.imshow('surf', img)
cv.waitKey(0)
# 输出surf是否检测特征点的方向
print(surf.getUpright())  # False
surf.setUpright(True)
# 重新计算
kp = surf.detect(img, None)
img2 = cv.drawKeypoints(gray, kp, None, (255, 0, 0), 4)
cv.imshow('surf2', img)
cv.waitKey(0)
# 检查描述子的维数
print(surf.descriptorSize())  # 64
surf.getExtended()  # False
# 这里将其设置为128维度
surf.setExtended(True)
kp, des = surf.detectAndCompute(img, None)
print(surf.descriptorSize())  # 128
print(des.shape)  # (N, 128)
