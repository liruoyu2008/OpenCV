'''KNN手写识别'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 读取手写数字图像（非常大的一张图2000*1000）
img = cv.imread('./images/digits.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 将该图切分为20*20的小图共5000张，水平切分100次，竖直切分50次
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# 将其组成一张图片阵列，共50*100张，每张20*20，shape=(100，50，20，20)
x = np.array(cells)

# 准备训练/测试数据各占一半（左右各一半）（每张小图展平为400像素）
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# 准备标签（10个类别）
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# 初始化knn，训练， 然后寻找测试数据的最近 k 个邻居
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

# 对比knn结果和测试图的实际标签，可得分类精度
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( accuracy )