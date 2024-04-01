'''KNN（K-最近邻算法）'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 特征点集包含25个（x,y）已知二维点作为训练数据
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# 每个点给标签 0/1 作为不同的 2 个类别，
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# 其中标签为 0 的点，绘制为红点
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# 标签为 1 的点，绘制为蓝点
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

plt.show()

# 一个用于预测的新的点
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)

print( "result:  {}\n".format(results) )
print( "neighbours:  {}\n".format(neighbours) )
print( "distance:  {}\n".format(dist) )

plt.show()