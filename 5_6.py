'''FAST（Features from Accelerated Segment Test）角点检测
虽然FAST算法本身不直接使用机器学习，但可以借助机器学习来优化其性能:
1.自动选择阈值：
问题：
在传统的FAST算法中，需要手动设置一个阈值来决定何种亮度差异足够大，
从而可以认定一个点为角点。这个阈值的选择对算法的性能有显著影响。
解决方案：
使用机器学习模型（如决策树、随机森林）来学习最优阈值。可以通过将图像的一部分用于训练，
其中包含已知的角点和非角点，模型学习如何基于局部图像特征自动确定最佳阈值。
2.优化角点检测：
问题：
原始的FAST算法可能会产生大量的角点，其中一些可能是噪声引起的，不是真正意义上的特征点。
解决方案：
利用机器学习对检测到的角点进行“筛选”和“打分”。例如，可以训练一个分类器来区分“好”的角点和“坏”的角点，
基于角点周围的像素模式。这样不仅减少了错误检测的角点数量，还提高了角点的质量。
'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./images/15.png',0)
# 使用默认参数初始化FAST对象
fast = cv.FastFeatureDetector_create()
# 寻找和呈现特征点
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# 打印出所有参数
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv.imshow('fast_true',img2)
cv.waitKey(0)
# 禁用非最大之抑制
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imshow('fast_false',img3)
cv.waitKey(0)
cv.destroyAllWindows()