'''图像分割（分水岭算法）'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('./images/18.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
fig, plots = plt.subplots(4, 4)
plots[0, 0].imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plots[0, 0].set_title('raw')

ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
plots[0, 1].imshow(thresh, cmap='gray')
plots[0, 1].set_title('otsu')

# 形态学去噪（开运算）
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
plots[0, 2].imshow(opening, cmap='gray')
plots[0, 2].set_title('opening')
# 查找确定的背景区域（扩大前景，使得所有边界一定被包含在内）
sure_bg = cv.dilate(opening, kernel, iterations=3)
plots[0, 3].imshow(sure_bg, cmap='gray')
plots[0, 3].set_title('sure_bg')
# 查找确定的前景区域
# 距离变换，用于求出前景中各点到最近的背景点的距离，距离越大，值越大。
# 常用于对象分割、形状分析
# # 此处使用了L2范式距离，即欧氏距离，也就是两点间的线段长度。
# 此外还有cv.DIST_L1：曼哈顿距离（A点到B点只走横、竖线的路线长度）
# cv.DIST_C：切比雪夫距离，横向、纵向距离取较大者。
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
plots[1, 0].imshow(dist_transform, cmap='gray')
plots[1, 0].set_title('dist_transform')
plots[1, 1].imshow(sure_fg, cmap='gray')
plots[1, 1].set_title('sure_fg')
# 查找不确定区域（上述二者作差集）
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
plots[1, 2].imshow(unknown, cmap='gray')
plots[1, 2].set_title('unknown')

# 标记标签
ret, markers = cv.connectedComponents(sure_fg)
# 给所有的标签+1，因为分水岭中，非确定区域应该记为0
markers = markers+1
# 将不确定区区域标记为0,并将各部分用渐变伪彩图呈现
markers[unknown==255] = 0
plots[1, 3].imshow(markers,cmap=plt.get_cmap('jet'))
plots[1, 3].set_title('markers')
# 应用分水岭算法
markers = cv.watershed(img,markers)
img[markers == -1] = [0,0,255]
plots[2, 0].imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plots[2, 0].set_title('watershed')

plt.show()
