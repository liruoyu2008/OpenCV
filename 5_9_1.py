'''BFMatch蛮力匹配（ORB）'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('./images/21.jpg',0)           # 查询图像
img2 = cv.imread('./images/22.jpg',0)           # 目标图像（训练图像）
# 初始化检测器
orb = cv.ORB_create()
# 查找ORB特征点和描述子
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# 创建BFMatcheer蛮力匹配器
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# 匹配描述子
matches = bf.match(des1,des2)
# 根据距离排序
matches = sorted(matches, key = lambda x:x.distance)
# 画出前10个匹配
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
plt.imshow(img3),plt.show()