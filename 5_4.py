'''SIFT特征
1.高斯金字塔（多个octave，一般4~5个）
2.每个octave内图像作不同模糊级别得到多张图像（总共n张，一般6张）
3.相邻模糊图作高斯差分DoG得到n-1张差分图（2和3共同实现尺度不变性）
4.差分图中间的n-3张，每张的每个像素点位置与其3D领域内的周围26个点比较看是否是极值
5.若时极值，则针对该点在该差分图周围邻域进行二次函数的拟合，求取极值点真实位置（不是像素坐标，而是实数坐标，亚像素级别）
6.多个octave内的多张差分图的所有找到的极值点合并，并考虑其位置（原始图像）、梯度方向（在找到该点的差分图对应的两张模糊图中较模糊的那一张中进行）对比度、边缘响应等情况进行筛选
7.求该位置在差分图对应的两张模糊图中较模糊的那一张的2西格玛（该图层的高斯模糊核的标准差，同一个octave内越模糊越大）半径的圆形邻域内的主方向作为该特征点方向
8.以该方向作为0度方向，在与上一步同样的图层中求该位置的16*16像素邻域内的梯度方向分布，作为其特征点描述子KPD（利用主方向和环境方向实现局部旋转不变性）。
9.所有的KPD组成一个N*128的矩阵返回（N个KPD，每个128维向量），并返回关键点信息（在原图中的位置、主方向、尺度等等信息）

'''
import numpy as np
import cv2 as cv

img = cv.imread('./images/15.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
# kp是关键点信息，des为KPD阵列（N*128）
kp, des = sift.detectAndCompute(gray,None)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sfit',img)
cv.waitKey(0)