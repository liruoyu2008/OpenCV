'''通过特征匹配+单应性，进行对象查找'''
'''RANSAC单应性变换方法的主要步骤：
1.随机选取：随机选取四对匹配点，计算透视变换矩阵。
2.应用变换：使用这个矩阵将所有输入点变换到目标图像空间。
3.计算一致性：计算多少点对的变换结果与其对应目标点足够接近（即距离小于某个阈值）。
4.迭代重复：重复上述过程多次，每次保留产生最多一致点对的变换矩阵。
5.使用最佳模型：使用找到的最佳变换矩阵，并可能使用所有被该模型认为是内点的点对，通过最小二乘法重新计算最终的透视变换矩阵以提高精度。
综上。本质上，RANSAC方法是通过随机选取4个匹配进行透视变换的计算，然后通过多次迭代寻找这些变换中拟合的最好的结果。
而所谓“内点”就是在对应的透视变换下，一致性值小于阈值的点，与之相反的就是“外点”'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('./images/21.jpg',0)           # 查询图像
img2 = cv.imread('./images/22.jpg',0)           # 目标图像（训练图像）

# 特征匹配
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
# 存储通过比率测试的匹配.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m) 
        
# 对象查找（仅在至少有10个合格匹配的请情况下进行）
MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w= img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # 绘制匹配
                   singlePointColor = None,
                   matchesMask = matchesMask, # 绘制轮廓
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()