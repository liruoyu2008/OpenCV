'''BRIEF（Binary Robust Independent Elementary Features）二值鲁棒独立基本特征'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./images/15.png',0)
# 初始化STAR特征检测器
star = cv.xfeatures2d.StarDetector_create()
# 初始化BRIEF描述子抽取对象
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# 检测STAR特征点
kp = star.detect(img,None)
# 计算特征点的BRIEF描述
kp, des = brief.compute(img, kp)
print( brief.descriptorSize() )
print( des.shape )