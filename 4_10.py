from configparser import Interpolation
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./images/Lenna.jpg')
fig, axs = plt.subplots(5, 4)
axs[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0, 0].set_title('raw')

# 使用pyplot直接绘制直方图(注意：彩图和灰度图得到的直方图分布是不一样的)
axs[0, 1].hist(img.ravel(), 256, [0, 256])
axs[0, 1].set_title('pyplot hist')

# 使用opencv计算直方图数据，然后使用数据绘图
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    axs[0, 2].plot(hist, color=col)
    axs[0, 2].set_xlim([0, 256])
    axs[0, 2].set_title('cv hist(3 channels)')

# 也可配合遮罩使用，用于对roi区域进行统计
mask = np.zeros(img.shape[:2], np.uint8)
mask[50:100, 50:100] = 255
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    axs[0, 3].plot(hist, color=col)
    axs[0, 3].set_xlim([0, 256])
    axs[0, 3].set_title('cv hist with mask(3 channels)')

# 利用直方图均衡，扩大直方图中主体部分像素的对比度。
# 代价：牺牲非主体部分的对比度。然而非主体部分像素数量一般很有限，不重要啦
img2 = cv.imread('./images/12.png', 0)
equ = cv.equalizeHist(img2)
axs[1, 0].imshow(img2, cmap='gray')
axs[1, 0].set_title('raw2')
axs[1, 1].imshow(equ, cmap='gray')
axs[1, 1].set_title('equalizeHist')

# 自适应直方图均衡（使用黄河讲课视频截图进行对比）
# 原图
img3 = cv.imread('./images/13.png', 0)
axs[2, 0].imshow(img3, cmap='gray')
axs[2, 0].set_title('raw3')
# 直方图
axs[2, 1].hist(img3.ravel(), 256, [0, 256])
axs[2, 1].set_title('raw3 hist')
# CLAHE图
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img3)
axs[2, 2].imshow(cl1, cmap='gray')
axs[2, 2].set_title('CLAHE')
# CLAHE直方图
axs[2, 3].hist(cl1.ravel(), 256, [0, 256])
axs[2, 3].set_title('CLAHE hist')

# 对彩色图进行CLAHE
# 读取图像
img4 = cv.imread('./images/13.png')
# 转换到 LAB 色彩空间
lab_image = cv.cvtColor(img4, cv.COLOR_BGR2LAB)
# 分离 L, A, B 通道
l, a, b = cv.split(lab_image)
# 创建CLAHE对象
clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
# 在 L 通道上应用CLAHE
cl = clahe.apply(l)
# 合并调整后的 L 通道和原始 A, B 通道
merged_lab = cv.merge((cl, a, b))
axs[3, 0].imshow(cv.cvtColor(img4,cv.COLOR_BGR2RGB))
axs[3, 0].set_title('raw4')
axs[3, 1].imshow(cv.cvtColor(merged_lab,cv.COLOR_Lab2RGB))
axs[3, 1].set_title('CLAHE')

# 2D直方图
img5 = cv.imread('./images/14.png')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# 统计亮度值、色度值两个维度，生成2维直方图
# 类似伪彩图，横纵坐标分别表示明度、色度，每个点的亮度才表示该明度、亮度的点的占比
hist5 = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
axs[4, 0].imshow(cv.cvtColor(img5,cv.COLOR_BGR2RGB))
axs[4, 0].set_title('raw5')
axs[4, 1].imshow(hist5,interpolation = 'nearest')
axs[4, 1].set_title('2d hist')

# 反投影。相当于通过直方图筛Hist_ROI，然后利用Hist_ROI反向计算原图的ROI，从而筛选原图像素
roi = cv.imread('./images/16.png')
hsv2 = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
target = cv.imread('./images/15.png')
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# 计算模板图像的二值化分布
roihist = cv.calcHist([hsv2],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# 归一化并应用反投影
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# 使用圆盘滤波器卷积（非必须，但能提高抠图质量）
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)
# 阈值化并蒙版抠图
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
res = np.vstack((target,thresh,res))
axs[4, 2].imshow(cv.cvtColor(res,cv.COLOR_BGR2RGB))
axs[4, 2].set_title('直方图反投影抠图')



plt.show()
