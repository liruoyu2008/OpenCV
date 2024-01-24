'''
Canny边缘检测，一个多阶段边缘检测算法，该算法旨在保留最少、最细、最清晰的边缘
1.降噪，使用一个5x5的高斯滤波器进行降噪
2.寻找图像梯度，这一步寻找图像中的边缘
3.非极大值抑制，图像在某一方向（算法分为8方向分别进行）某一邻域的变化仅取其最剧烈的部分，忽略掉微小的渐变的特征
4.滞后阈值，明显的边缘被忽略，明显的边缘被保留，其他边缘根据与“明显边缘”的连通性确定是否保留
'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./images/Lenna.jpg',0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()