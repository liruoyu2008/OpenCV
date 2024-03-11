'''角点检测'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 哈里斯角点检测
filename = './images/9.png'
img = cv.imread(filename)
fig, plots = plt.subplots(2, 2)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plots[0, 0].imshow(gray, cmap='gray')
plots[0, 0].set_title('gray')
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
# 将角点的标记放大呈现
dst = cv.dilate(dst, None)
# 选取合适的阈值，阈值的最优解依赖于具体的图像
img[dst > 0.6*dst.max()] = [0, 0, 255]
plots[0, 1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plots[0, 1].set_title('harris')

# 亚像素级角点检测
dst = np.uint8(dst)
# 寻找质心
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# 对角点位置进行亚像素级精细化
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
# 画出结果
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0, 0, 255]
img[res[:, 3], res[:, 2]] = [0, 255, 0]
plots[1, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plots[1, 0].set_title('subpix')
plt.show()
