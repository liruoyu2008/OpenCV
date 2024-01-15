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
    '''仿射变换（三角形映射法）'''
    img = cv.imread('Lenna.jpg')
    rows, cols, ch = img.shape
    # 一般可选取图像的角点作锚点
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('input', img)
    cv.imshow('output', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def func4():
    '''仿射变换（变换矩阵法）'''
    img = cv.imread('Lenna.jpg')
    rows, cols, ch = img.shape
    # 绕原点（左上角点）向角度正方向（与笛卡尔逆时针方向不同的是，由于y轴朝下，opencv2D图形是顺时针）旋转60度，然后向x、y正方向分别移动200、30
    M = np.array([[np.cos(np.pi/3),np.sin(-np.pi/3),200],[np.sin(np.pi/3),np.cos(np.pi/3),30]])
    dst = cv.warpAffine(img, M, (cols, rows))
    plt.subplot(121), plt.imshow(cv.cvtColor(
        img, code=cv.COLOR_BGR2RGB)), plt.title('input')
    plt.subplot(122), plt.imshow(cv.cvtColor(
        dst,  code=cv.COLOR_BGR2RGB)), plt.title('output')
    plt.show()


def func5():
    '''透视变换（四边形映射法）'''
    img = cv.imread('Lenna.jpg')
    rows, cols, ch = img.shape
    # 选取图像的角点作锚点
    pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts2 = np.float32([[0, 0], [600, 52], [0, 800], [600, 320]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    M2 = cv.getPerspectiveTransform(pts2, pts1)
    # 模拟得到透视效果
    dst = cv.warpPerspective(img, M, (800, 800))
    # 模拟通过透视还原图像
    src2 = cv.warpPerspective(dst, M2, (400, 400))
    plt.subplot(221), plt.imshow(cv.cvtColor(
        img, code=cv.COLOR_BGR2RGB)), plt.title('input')
    plt.subplot(222), plt.imshow(cv.cvtColor(
        dst,  code=cv.COLOR_BGR2RGB)), plt.title('output')
    plt.subplot(223), plt.imshow(cv.cvtColor(
        src2,  code=cv.COLOR_BGR2RGB)), plt.title('output2')
    plt.show()


if __name__ == '__main__':
    func5()
