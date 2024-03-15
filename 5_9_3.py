'''FLANN快速最近邻搜索包匹配（Fast Library for Approximate Nearest Neighbors）（SIFT）
原理：BF是将每个A特征与每个B特征进行匹配，时间复杂度为O(n^2).FLANN在BF的基础上，通过构造一定的
数据结构(例如KDTree、KMeansTree等)，加快从A特征搜索B特征的过程，时间复杂度通常近似认为O(log n)。
可以类比二叉排序树的排序原理。
相对于BF蛮力搜索，牺牲精度，提高效率，适用于大规模数据集'''
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
# FLANN 参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)     # 或传递一个空字典
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# 只需画出好的匹配，所以创建一个掩膜
matchesMask = [[0,0] for i in range(len(matches))]
# 比率测试
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()