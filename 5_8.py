'''ORB（Oriented FAST and Rotated BRIEF）定向FAST和轮转BRIEF'''
# ORB相对于FAST和BRIEF的关键改进点
# 旋转不变性：相比于原始的BRIEF，ORB通过为每个特征点计算主方向并相应旋转BRIEF描述子的采样模式，引入了旋转不变性。
# 尺度不变性：通过在尺度金字塔上应用FAST特征点检测，ORB增加了对尺度变化的鲁棒性。
# 高效性：ORB继承了FAST和BRIEF的高效计算性能，使其适合实时应用。
# 特征点选择：ORB在特征点过多时，使用快速排序的方式选取前N个最优的特征点，依据是特征点的Harris角响应值，这一步骤保证了特征点的质量。
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./images/7.png', 0)
# 初始化ORB检测器
orb = cv.ORB_create()
# 检测ORB特征点
kp = orb.detect(img, None)
# 计算描述子
kp, des = orb.compute(img, kp)
# 画出特征点（不展示特征点尺度和方向）
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
plt.imshow(img2), plt.show()
