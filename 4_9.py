'''
opencv轮廓
图像的轮廓及其矩是关于物体外形的非常重要的特征
根据不同矩的多项式组合
（例如，M20表示图像的沿x轴的2阶矩即波动情况;M20-2M11+M02表示图像沿2阶原点矩用于估算图像沿x轴的波动情况），
可以求取图像的质心（可能是视觉重点）、任意方向的波动情况（中心矩）等重要分布情况
这几乎可以近乎精确的估出源图像的全部形状信息（但是计算是不可能穷尽的）

'''

from importlib import import_module
from operator import index
from tarfile import RECORDSIZE
import cv2 as cv
from matplotlib.dviread import Box
import numpy as np
from matplotlib import pyplot as plt


raw = cv.imread('./images/8.png')
gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# 画出轮廓
# 轮廓本身并不是一幅图，而是一串结构化数据，记录了所有轮廓点的坐标，因此想展现轮廓，必须用轮廓对应的方法函数，如cv.drawContours
cont_img = raw.copy()
cont = contours[0]
cv.drawContours(cont_img, [cont], 0, (0, 0, 255), 2)
# 求轮廓的矩
moments = cv.moments(cont)
(m00, m10, m01, m20, m11, m02) = (moments['m00'], moments['m10'], moments['m01'], moments['m20'], moments['m11'], moments['m02'])
# 面积
area = m00
area_2 = cv.contourArea(cont)
# 周长
area_C = cv.arcLength(cont, True)
# 质心
(Mx, My) = (int(m10/m00), int(m01/m00))
# 将质心画在原图中
m_img = raw.copy()
cv.circle(m_img, (Mx, My), 2, (0, 0, 255), cv.FILLED)
# 凸包
hull_img = raw.copy()
k = cv.isContourConvex(cont)
hull = cv.convexHull(cont)
hull_img = cv.drawContours(hull_img, [hull], 0, (0, 0, 255), 2)
# 边界矩形、外接矩形、外接圆
graph_img = raw.copy()
bouding = cv.boundingRect(cont)
cv.rectangle(graph_img, bouding, (150, 0, 0), 2)
rect = cv.minAreaRect(cont)
box = cv.boxPoints(rect)
box = np.int0(box)  # int0是平台整数（Platform Integer）,所以此处实际上等效于进行int64的转换
# box = np.int64(box)
graph_img = cv.drawContours(graph_img, [box], 0, (0, 150, 0), 2)    # 由于最小外接矩形是经过旋转的矩形，`因此不能用cv.rectangle绘制，而因该作为轮廓绘制cv.drawcontour
circle = cv.minEnclosingCircle(cont)
ellipse = cv.fitEllipse(cont)
triangle = cv.minEnclosingTriangle(cont)
point = tuple(int(x) for x in circle[0])
cv.circle(graph_img, point, int(circle[1]), (0, 0, 150), 2)
cv.ellipse(graph_img, ellipse, (0, 255, 0), 2)
cv.polylines(graph_img, [triangle[1].astype(np.int32)], True, (100, 100, 0), 2)
# 将轮廓的DP（道格拉斯-普克抽稀算法）结果画出
cont_approx_img_1 = raw.copy()
cont_approx_img_2 = raw.copy()
e1 = 0.1*area_C
e2 = 0.01*area_C
cont_approx_1 = cv.approxPolyDP(cont, e1, True)
cont_approx_2 = cv.approxPolyDP(cont, e2, True)
cv.drawContours(cont_approx_img_1, [cont_approx_1], 0, (0, 0, 255), 2)
cv.drawContours(cont_approx_img_2, [cont_approx_2], 0, (0, 0, 255), 2)
# 展示
dict = {
    'raw': raw,
    'cont': cont_img,
    'M_center': m_img,
    'cont_approx_1': cont_approx_img_1,
    'cont_approx_2': cont_approx_img_2,
    'hull': hull_img,
    'graph': graph_img,
}
index = 1
for k, v in dict.items():
    plt.subplot(3, 3, index)
    plt.title(k)
    plt.axis('off')
    plt.imshow(cv.cvtColor(v, cv.COLOR_BGR2RGB))
    index += 1
x = cont[cont[:, :, 0].argmin()]
plt.show()


# 此外，轮廓还有很多属性及:
# 宽高比：轮廓的边界矩形（非最小外接矩形）的宽高比，使用cv.boundingRect()获取边界矩形（另可使用cv.minAreaRect得到最小外接矩形）
# 范围：轮廓与其边界矩形的区域比率
# 固实性：轮廓与其凸包的面积比率,使用cv.isContourConvex()判断凸度，cv.convexHull()求凸包
# 等效直径：面积与轮廓面积相等的圆的直径
# 方向：主轴方向，可以使用椭圆或直线拟合得到方向
# 遮罩和像素点：可以使用轮廓生成遮罩，并使用bitwise_and在原图中筛选对应区域的像素
# 最大值、最小值及其位置:cv.minMaxLoc函数求图像（或遮罩对应的部分）中的像素最值及其位置
# 平均值（灰度强度或颜色）：cv.mean
# 极端点：图像的最顶部、最底部、最左侧、最右侧的点

# 轮廓的功能：
# 检测凸包缺陷：
hull = cv.convexHull(cont, returnPoints=False)
defects = cv.convexityDefects(cont, hull)
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cont[s][0])
    end = tuple(cont[e][0])
    far = tuple(cont[f][0])
    cv.line(raw, start, end, [0, 255, 0], 2)
    cv.circle(raw, far, 5, [0, 0, 255], -1)
cv.imshow('img', raw)
cv.waitKey(0)
cv.destroyAllWindows()
# 点到轮廓的距离（内正外负）：
dist = cv.pointPolygonTest(cont, (50, 50), True)
# 形状匹配(利用Hu-Moment胡矩),结果越小越匹配：
img1 = cv.imread('./images/9.png', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread('./images/10.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./images/11.png', cv.IMREAD_GRAYSCALE)
ret, thresh = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, 2, 1)
cnt1 = contours[0]
contours, hierarchy = cv.findContours(thresh2, 2, 1)
cnt2 = contours[0]
ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
input('任意键退出...')
