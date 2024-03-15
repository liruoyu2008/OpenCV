'''BRIEF（Binary Robust Independent Elementary Features）二值鲁棒独立基本特征'''
'''BRIEF的核心思想是通过某种特征检测手段检测到特征点集kp和特征描述子集合des，然后对于
kp内N个特征点的周围领域内的信息采用自己的方式进行重新抽取，得到一个128/256/512位（16/32/64字节）的数据。
由于原始的des信息对于BRIEF是没用的，所以很多侧重点在描述子上的特征检测算法对于BRIEF其实必要性不大，加之特征点其实是
图像本身的固有信息，不同的特征检测算法只是不用程度的去发现和采集这些点而已，对于成熟的特征检测算法，检测到的特征点大多是相似的，
因此，BRIEF的特征检测过程更青睐快速高效的检测算法。
新的描述过程最终会将N*128*64Byte的数据压缩为N*(16/32/64)Byte的数据，这样可以便于进行后续的特征匹配过程'''
# BRIEF将像素点对坍缩成一个bit的过程具体是如何进行的？
# BRIEF算法通过以下步骤将特征点周围领域的信息坍缩成一个二进制位（bit）：
# 领域采样：对于给定的特征点，BRIEF首先在其周围定义一个领域（例如一个正方形区域）。
# 然后，在这个领域内随机选择像素点对。这些点对的位置可以事先定义，或者每次运行时随机生成。
# 亮度比较测试：对于每一对随机选择的像素点# (p,q)，BRIEF比较它们的亮度值。如果点p的亮度值大于点q的亮度值，
# 则该比较的结果为1；否则，结果为0。这个步骤产生了一个二进制位。
# 生成描述子：重复上述亮度比较测试多次（例如256次），每次比较生成一个二进制位。将这些二进制位串联起来，
# 形成一个二进制字符串。对于256次测试，这会生成一个256位的二进制描述子。
# 这个过程生成的二进制描述子紧凑而高效，可以快速进行特征点匹配。比较两个描述子之间的相似度通常使用汉明距离（Hamming distance），
# 即两个二进制字符串不同位的数量。# 通过这种方式，BRIEF能够以非常高效的方式捕获图像中特征点周围的局部纹理信息，
# 并将这些信息压缩成一个简洁的二进制形式。这使得BRIEF非常适合于在性能受限的环境中进行实时图像匹配任务。
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