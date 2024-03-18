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
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
# 在 L 通道上应用CLAHE
cl = clahe.apply(l)
# 合并调整后的 L 通道和原始 A, B 通道
merged_lab = cv.merge((cl, a, b))
axs[3, 0].imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
axs[3, 0].set_title('raw4')
axs[3, 1].imshow(cv.cvtColor(merged_lab, cv.COLOR_Lab2RGB))
axs[3, 1].set_title('CLAHE')

# 2D直方图
img5 = cv.imread('./images/14.png')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# 统计亮度值、色度值两个维度，生成2维直方图
# 类似伪彩图，横纵坐标分别表示明度、色度，每个点的亮度才表示该明度、亮度的点的占比
# 参数和原理：
# channels--每幅图像的通道数。重要。该数量是直方图统计时需要考虑的维度，例如要统计色相H和饱和度S，那么对于
# images--输入图像的数量。不重要。可以为任意，但不同的图像的同一通道会合并进行统计HSV图像来说，每幅图像就需要指定两个通道[0, 1]进行统计，每个通道对应一个统计维度
# ranges--统计范围。从数值上规定统计应该考虑的像素值的范围，不在该范围内的像素直接忽略。每个channel应该提供2个参数值（2个值构成一个范围）。
# mask--掩膜。特殊。从空间上精细控制进入统计范围的像素的位置分布。
# histsize--直方图尺寸。参数个数与通道数一致，并且其值最终与返回结果的shape一致。控制了统计的颗粒度或精细程度。表示将某channel的的统计范围平均分为多少个bin（箱子/盒子，用于表示一个小的范围），落入某一bin对应的像素值会在使得该bin的统计值+1。
hist5 = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
axs[4, 0].imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
axs[4, 0].set_title('raw5')
axs[4, 1].imshow(hist5, interpolation='nearest')
axs[4, 1].set_title('2d hist')

# 反投影。相当于通过直方图筛Hist_ROI，然后利用Hist_ROI反向计算原图的ROI，从而筛选原图像素
# 原理：例如若A为模板图，B为目标图，C为反投影结果图，那么B、C尺寸一致，且C实际上是一张概率分布图，每个像素点表示一个概率值。
# 若模板A的直方图（直方图本质就是频率或概率分布图）代表的概率分布中仅有90%的绿色像素和10%的红色像素，
# 那么B中任何位置的绿色像素都将在C的同等位置显示较高的值，而B中任何位置的红色像素都将显示较低的值。
# 因此，直方图方向投影的意义实质上是对B中所有像素注意评估可能存在于模板A中的概率大小。利用结果图C来计算掩膜并用适当的
# 阈值进行二值化可以得到能筛选出这部分像素的掩膜。
roi = cv.imread('./images/16.png')
hsv2 = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
target = cv.imread('./images/15.png')
hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
# 计算模板图像的二值化分布
roihist = cv.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
# 归一化并应用反投影
# 归一化的必要性：2D概率分布直方图中的每个点并不是像素值而是频率值。但是，一旦考察到hsvt中某像素落在直方图的某位置上，反向投影为了计算简便
# 会直接将直方图的频率值赋值到dst的该位置上。因此，若需要将dst作为图像看待的话，归一化必不可少。
# 当然，如果归一化不是对roihist进行，而是对dst进行，实际得到的dst也是一样的。
cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
# 使用圆盘滤波器卷积（非必须，但能提高抠图质量）
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
dst = cv.filter2D(dst, -1, disc, dst)
# 阈值化并蒙版抠图
ret, thresh = cv.threshold(dst, 50, 255, 0)
thresh = cv.merge((thresh, thresh, thresh))
res = cv.bitwise_and(target, thresh)
res = np.vstack((target, thresh, res))
axs[4, 2].imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
axs[4, 2].set_title('calcBack')


plt.show()
