'''BFMatch蛮力匹配（SIFT）'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('./images/21.jpg',0)           # 查询图像
img2 = cv.imread('./images/22.jpg',0)           # 目标图像（训练图像）
# 初始化SIFT检测器
sift = cv.xfeatures2d.SIFT_create()
# 找到SIFT特征点和描述符
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# 使用默认参数进行BFMatch
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# 进行比率测试
good = []
# 解包双层列表，读取到k=2的两个最邻近匹配，
# 最佳匹配与次佳匹配二者相差越大，说明最佳匹配越显著越明确，因而越可靠
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# 画出筛选出的最佳匹配
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
plt.imshow(img3),plt.show()