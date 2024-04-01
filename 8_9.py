'''K-Means聚类'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

X = np.random.randint(10,50,(25,2))
Y = np.random.randint(50,90,(25,2))
Z = np.vstack((X,Y))

Z = np.float32(Z)

# 定义停止条件，并应用聚类
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# 分割数据，注意将其扁平化
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]

# 绘制点
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()