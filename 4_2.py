import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def func1():
    '''缩放'''
    img = cv.imread('1.jpg')
    res = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    cv.imshow("raw", img)
    cv.imshow("red", res)
    key = cv.waitKey(5000)


def func2():
    '''仿射变换(透视变换)'''
    img = cv.imread('1.jpg', 0)
    rows, cols = img.shape
    
    # 以下仿射矩阵 θ=0，δ=^[100，50],因此仅起到平移作用
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def func3():
    '''旋转'''
    img = cv.imread('1.jpg',0)
    rows,cols = img.shape
    # 生成仿射矩阵，cols-1 和 rows-1 是坐标限制
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,0.5)
    dst = cv.warpAffine(img,M,(cols,rows))
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def func4():
    '''三角仿射'''
    img = cv.imread('1.jpg')
    rows,cols,ch = img.shape
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img,M,(800,800))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()


if __name__=='__main__':
    func4()