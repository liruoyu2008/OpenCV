'''图像阈值'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def func1():
    '''渐变图像全局（常量）阈值'''
    img = cv.imread('./images/gradient.jpg')
    ret, thr1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thr2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thr3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thr4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thr5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    cv.imshow('raw', img)
    cv.imshow('BINARY', thr1)
    cv.imshow('BINARY_INV', thr2)
    cv.imshow('TRUNC', thr3)
    cv.imshow('TOZERO', thr4)
    cv.imshow('TOZERO_INV', thr5)
    cv.waitKey(0)


def func2():
    '''全局（常量）阈值处理'''
    img = cv.imread('./images/gradient.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thr_val, img_ret11 = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY)
    thr_val, img_ret12 = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
    thr_val, img_ret13 = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV)
    thr_val, img_ret21 = cv.threshold(img_gray, 127, 255, cv.THRESH_TRUNC)
    thr_val, img_ret22 = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO)
    thr_val, img_ret23 = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO_INV)
    thr_val, img_ret31 = cv.threshold(
        img_gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    thr_val, img_ret32 = cv.threshold(
        img_gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    # 显示图像
    fig, ax = plt.subplots(3, 3)
    ax[0, 0].set_title('THRESH_BINARY,50', fontsize=10)
    # matplotlib显示图像为rgb格式
    ax[0, 0].imshow(cv.cvtColor(img_ret11, cv.COLOR_BGR2RGB))
    ax[0, 1].set_title('THRESH_BINARY,127', fontsize=10)
    ax[0, 1].imshow(cv.cvtColor(img_ret12, cv.COLOR_BGR2RGB))
    ax[0, 2].set_title('THRESH_BINARY_INV,200', fontsize=10)
    ax[0, 2].imshow(cv.cvtColor(img_ret13, cv.COLOR_BGR2RGB))
    ax[1, 0].set_title('THRESH_TRUNC,127', fontsize=10)
    ax[1, 0].imshow(cv.cvtColor(img_ret21, cv.COLOR_BGR2RGB))
    ax[1, 1].set_title('THRESH_TOZERO,127', fontsize=10)
    ax[1, 1].imshow(cv.cvtColor(img_ret22, cv.COLOR_BGR2RGB))
    ax[1, 2].set_title('THRESH_TOZERO_INV,127', fontsize=10)
    ax[1, 2].imshow(cv.cvtColor(img_ret23, cv.COLOR_BGR2RGB))
    ax[2, 0].set_title('THRESH_OTSU', fontsize=10)
    ax[2, 0].imshow(cv.cvtColor(img_ret31, cv.COLOR_BGR2RGB))
    ax[2, 1].set_title('THRESH_TRIANGLE', fontsize=10)
    ax[2, 1].imshow(cv.cvtColor(img_ret32, cv.COLOR_BGR2RGB))
    ax[2, 2].set_title('RAW', fontsize=10)
    ax[2, 2].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[1, 2].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[2, 2].axis('off')
    plt.show()


def func3():
    '''自适应阈值处理'''
    img = cv.imread('./images/Lenna.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thr_val, img_ret11 = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY)
    thr_val, img_ret11 = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C)
    thr_val, img_ret31 = cv.threshold(
        img_gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    thr_val, img_ret32 = cv.threshold(
        img_gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    # 显示图像
    fig, ax = plt.subplots(3, 3)
    ax[0, 0].set_title('THRESH_BINARY,50', fontsize=10)
    # matplotlib显示图像为rgb格式
    ax[0, 0].imshow(cv.cvtColor(img_ret11, cv.COLOR_BGR2RGB))
    ax[0, 1].set_title('THRESH_BINARY,127', fontsize=10)
    ax[0, 1].imshow(cv.cvtColor(img_ret12, cv.COLOR_BGR2RGB))
    ax[0, 2].set_title('THRESH_BINARY_INV,200', fontsize=10)
    ax[0, 2].imshow(cv.cvtColor(img_ret13, cv.COLOR_BGR2RGB))
    ax[1, 0].set_title('THRESH_TRUNC,127', fontsize=10)
    ax[1, 0].imshow(cv.cvtColor(img_ret21, cv.COLOR_BGR2RGB))
    ax[1, 1].set_title('THRESH_TOZERO,127', fontsize=10)
    ax[1, 1].imshow(cv.cvtColor(img_ret22, cv.COLOR_BGR2RGB))
    ax[1, 2].set_title('THRESH_TOZERO_INV,127', fontsize=10)
    ax[1, 2].imshow(cv.cvtColor(img_ret23, cv.COLOR_BGR2RGB))
    ax[2, 0].set_title('THRESH_OTSU', fontsize=10)
    ax[2, 0].imshow(cv.cvtColor(img_ret31, cv.COLOR_BGR2RGB))
    ax[2, 1].set_title('THRESH_TRIANGLE', fontsize=10)
    ax[2, 1].imshow(cv.cvtColor(img_ret32, cv.COLOR_BGR2RGB))
    ax[2, 2].set_title('RAW', fontsize=10)
    ax[2, 2].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[1, 2].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[2, 2].axis('off')
    plt.show()


if __name__ == '__main__':
    func1()
