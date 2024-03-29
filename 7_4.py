'''立体图像的深度'''
# 基本上，如果你有两个摄像头的基础矩阵（Fundamental Matrix），就可以利用立体视觉的原理
# 来估算场景中物体的深度信息。通过两个视角拍摄到的同一场景的画面，结合摄像头间的几何关系，
# 可以计算出画面中物体的三维位置。这个过程通常涉及到对两个视图中的特征点进行匹配，然后通过三角测量方法
# 来确定这些点在三维空间中的位置。尽管实际应用可能会更复杂，但这是立体视觉和三维重建的基本原理。

# 对于非特征点的像素，StereoBM通过搜索匹配的块或窗口来寻找对应点。算法会在另一张图中搜索最佳匹配的像素块，
# 以此确定每个像素的视差。视差是指同一物体在两张图像中的位置差异，通过视差可以计算出该点的深度信息。
# StereoBM适用于所有像素点，不仅仅是特征点，从而生成整个场景的深度图。
# 具体来说，StereoBM算法通过比较两张图像中的相应区域或“块”来找到最佳匹配的像素块。这个过程涉及到
# 在一个预设的搜索范围内，对每个像素点在另一张图像中进行搜索，以找到最相似的像素块。这种相似性是通过计算
# 不同块之间的相似度指标（如：块匹配算法中常用的平方差和(SSD)、归一化交叉相关(NCC)等）来评估的。
# 找到最相似的块后，通过计算这两个块在两张图像中的位置差异（即视差）来估计每个像素的深度信息。
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('./images/my_left.jpg',0)
imgR = cv.imread('./images/my_right.jpg',0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()