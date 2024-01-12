import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def func1():
    '''缩放'''
    img = cv.imread('Lenna.jpg')
    res = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    cv.imshow("raw", img)
    cv.imshow("red", res)
    key = cv.waitKey(5000)


def func2():
    '''旋转'''
    img = cv.imread('Lenna.jpg', 0)
    rows, cols = img.shape
    # 生成仿射矩阵，cols-1 和 rows-1 是坐标限制
    M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 0.5)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def func3():
    '''仿射变换（双区域映射法）'''
    img = cv.imread('Lenna.jpg')
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def func4():
    '''透视变换（三角透视、三区域映射法）'''
    img = cv.imread('Lenna.jpg')
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (800, 800))
    plt.subplot(121), plt.imshow(cv.cvtColor(
        img, code=cv.COLOR_BGR2RGB)), plt.title('Input')
    plt.subplot(122), plt.imshow(cv.cvtColor(
        dst,  code=cv.COLOR_BGR2RGB)), plt.title('Output')
    plt.show()


if __name__ == '__main__':
    func4()
