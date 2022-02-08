'''形态变换'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('1.jpg')
kernel = np.ones((5, 5), np.uint8)


def erode():
    '''腐蚀'''
    return cv.erode(img, kernel, iterations=1)


def dilate():
    '''膨胀'''
    return cv.dilate(img, kernel, iterations=1)


def open():
    '''开运算（先腐蚀后膨胀，消除白点、噪声）'''
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)


def close():
    '''开运算（先腐蚀后膨胀，消除白点、噪声）'''
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)


def grad():
    '''形态学梯度'''
    return cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)


def top_hat():
    '''顶帽运算'''
    return cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)


def black_hat():
    '''黑帽运算'''
    return cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)


if __name__ == '__main__':
    plt.subplot(331), plt.imshow(img), plt.title('origin')
    plt.subplot(332), plt.imshow(erode()), plt.title('erode')
    plt.subplot(333), plt.imshow(dilate()), plt.title('dilate')
    plt.subplot(334), plt.imshow(open()), plt.title('open')
    plt.subplot(335), plt.imshow(close()), plt.title('close')
    plt.subplot(336), plt.imshow(grad()), plt.title('grad')
    plt.subplot(337), plt.imshow(top_hat()), plt.title('top_hat')
    plt.subplot(338), plt.imshow(black_hat()), plt.title('top_hat')

    plt.show()
