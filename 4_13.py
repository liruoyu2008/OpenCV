'''霍夫变换'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def func1():
    '''霍夫变换'''
    # 对于二值图像的每个亮点对，其对应得的ro值确定，但是theta值却可能有180种（假设直线检测需要精确到1度）
    # 那么这180个点就都会记一票。但如果图像中真的存在某条直线，它必然经过至少两个点A和B，
    # 那么A和B（A、B为二值图像的点）这两个点的连线将是一个确定的ro，theta值（假定为m,n），
    # 那么(m,n)这个点（霍夫空间的点）就会因为A和B而被共计2票。
    # 因此，得票数也反映了直线经过的点数。如果点数很多，就可以判定是否是实际存在的直线。
    # 换言之，霍夫变换只能检测直线而非线段，
    # 并且可能存在在某种阈值下，将多条短线段记为一条直线而忽视一条更长的线段
    img = cv.imread(cv.samples.findFile('./images/7.png'))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    cv.imshow('canny', edges)
    cv.waitKey(0)
    # 经过霍夫变换得到的直线集合（若阈值设置太高，可能会得到None哦）
    lines = cv.HoughLines(edges, 1, np.pi/180, 100)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('HoughLines', img)
    cv.waitKey(0)


def func2():
    '''概率霍夫变换（检测线段）'''
    img = cv.imread(cv.samples.findFile('./images/7.png'))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv.imshow('HoughLinesP', img)
    cv.waitKey(0)


if __name__ == "__main__":
    func2()
