'''形态变换'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike

img1 = cv.imread('./images/4.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./images/5.png', cv.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)


def raw(img: MatLike, *args) -> MatLike:
    '''返回原图'''
    return img


def erode(img: MatLike, kernel: MatLike) -> MatLike:
    '''腐蚀'''
    return cv.erode(img, kernel, iterations=1)


def dilate(img: MatLike, kernel: MatLike) -> MatLike:
    '''膨胀'''
    return cv.dilate(img, kernel, iterations=1)


def open(img: MatLike, kernel: MatLike) -> MatLike:
    '''开运算（先腐蚀后膨胀，消除高频白噪声即白点）'''
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)


def close(img: MatLike, kernel: MatLike) -> MatLike:
    '''闭运算（先膨胀后腐蚀，消除高频黑噪声即黑斑）'''
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)


def grad(img: MatLike, kernel: MatLike) -> MatLike:
    '''形态学梯度（膨胀和腐蚀之差，凸显轮廓和噪点）'''
    return cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)


def top_hat(img: MatLike, kernel: MatLike) -> MatLike:
    '''顶帽运算（开运算与原图差,突出高频白噪声）'''
    return cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)


def black_hat(img: MatLike, kernel: MatLike) -> MatLike:
    '''黑帽运算（闭运算与原图差，突出高频黑噪声）'''
    return cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)


if __name__ == '__main__':
    imgs = [img1, img2]
    funcs = [raw, erode, dilate, open, close, grad, top_hat, black_hat]
    index = 1
    for img in imgs:
        for func in funcs:
            plt.subplot(4, 4, index)
            plt.imshow(cv.cvtColor(func(img, kernel), cv.COLOR_BGR2RGB))
            plt.title(func.__name__, fontsize=6)
            plt.axis("off")
            index += 1
    plt.show()
