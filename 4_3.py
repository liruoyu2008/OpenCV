from matplotlib.transforms import offset_copy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def func1():
    '''渐变图像全局（常量）阈值'''
    img = cv.imread('./images/gradient.jpg', cv.IMREAD_GRAYSCALE)
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
    '''针对彩色图像进行阈值处理的测试(与对其各通道进行阈值化然后合并的效果一样)'''
    img = cv.imread('./images/Lenna.jpg')
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()
    # 彩色图像直接阈值化
    ret, thr1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thr2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thr3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thr4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thr5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    # 将通道阈值化后合并
    ret, merge_1_B = cv.threshold(B, 127, 255, cv.THRESH_BINARY)
    ret, merge_1_G = cv.threshold(G, 127, 255, cv.THRESH_BINARY)
    ret, merge_1_R = cv.threshold(R, 127, 255, cv.THRESH_BINARY)
    merge_1 = cv.merge([merge_1_B, merge_1_G, merge_1_R])
    ret, merge_2_B = cv.threshold(B, 127, 255, cv.THRESH_BINARY_INV)
    ret, merge_2_G = cv.threshold(G, 127, 255, cv.THRESH_BINARY_INV)
    ret, merge_2_R = cv.threshold(R, 127, 255, cv.THRESH_BINARY_INV)
    merge_2 = cv.merge([merge_2_B, merge_2_G, merge_2_R])
    ret, merge_3_B = cv.threshold(B, 127, 255, cv.THRESH_TRUNC)
    ret, merge_3_G = cv.threshold(G, 127, 255, cv.THRESH_TRUNC)
    ret, merge_3_R = cv.threshold(R, 127, 255, cv.THRESH_TRUNC)
    merge_3 = cv.merge([merge_3_B, merge_3_G, merge_3_R])
    ret, merge_4_B = cv.threshold(B, 127, 255, cv.THRESH_TOZERO)
    ret, merge_4_G = cv.threshold(G, 127, 255, cv.THRESH_TOZERO)
    ret, merge_4_R = cv.threshold(R, 127, 255, cv.THRESH_TOZERO)
    merge_4 = cv.merge([merge_4_B, merge_4_G, merge_4_R])
    ret, merge_5_B = cv.threshold(B, 127, 255, cv.THRESH_TOZERO_INV)
    ret, merge_5_G = cv.threshold(G, 127, 255, cv.THRESH_TOZERO_INV)
    ret, merge_5_R = cv.threshold(R, 127, 255, cv.THRESH_TOZERO_INV)
    merge_5 = cv.merge([merge_5_B, merge_5_G, merge_5_R])
    # 绘图
    fig, plots = plt.subplots(2, 6)
    plots[0, 0].set_title("raw", fontsize=6)
    plots[0, 0].axis("off")
    plots[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plots[0, 1].set_title("raw_BINARY", fontsize=6)
    plots[0, 1].axis("off")
    plots[0, 1].imshow(cv.cvtColor(thr1, cv.COLOR_BGR2RGB))
    plots[0, 2].set_title("raw_BINARY_INV", fontsize=6)
    plots[0, 2].axis("off")
    plots[0, 2].imshow(cv.cvtColor(thr2, cv.COLOR_BGR2RGB))
    plots[0, 3].set_title("raw_TRUNC", fontsize=6)
    plots[0, 3].axis("off")
    plots[0, 3].imshow(cv.cvtColor(thr3, cv.COLOR_BGR2RGB))
    plots[0, 4].set_title("raw_TOZERO", fontsize=6)
    plots[0, 4].axis("off")
    plots[0, 4].imshow(cv.cvtColor(thr4, cv.COLOR_BGR2RGB))
    plots[0, 5].set_title("raw_TOZERO_INV", fontsize=6)
    plots[0, 5].axis("off")
    plots[0, 5].imshow(cv.cvtColor(thr5, cv.COLOR_BGR2RGB))
    plots[1, 1].set_title("merge_BINARY", fontsize=6)
    plots[1, 1].axis("off")
    plots[1, 1].imshow(cv.cvtColor(merge_1, cv.COLOR_BGR2RGB))
    plots[1, 2].set_title("merge_BINARY_INV", fontsize=6)
    plots[1, 2].axis("off")
    plots[1, 2].imshow(cv.cvtColor(merge_2, cv.COLOR_BGR2RGB))
    plots[1, 3].set_title("merge_TRUNC", fontsize=6)
    plots[1, 3].axis("off")
    plots[1, 3].imshow(cv.cvtColor(merge_3, cv.COLOR_BGR2RGB))
    plots[1, 4].set_title("merge_TOZERO", fontsize=6)
    plots[1, 4].axis("off")
    plots[1, 4].imshow(cv.cvtColor(merge_4, cv.COLOR_BGR2RGB))
    plots[1, 5].set_title("merge_TOZERO_INV", fontsize=6)
    plots[1, 5].axis("off")
    plots[1, 5].imshow(cv.cvtColor(merge_5, cv.COLOR_BGR2RGB))
    plots[1, 0].axis("off")
    plt.show()


def func3():
    '''全局（常量）阈值处理'''
    img = cv.imread('./images/Lenna.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thr_val, img_ret11 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    thr_val, img_ret12 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    thr_val, img_ret13 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
    thr_val, img_ret21 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    thr_val, img_ret22 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    thr_val, img_ret23 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    thr_val, img_ret31 = cv.threshold(
        img, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    thr_val, img_ret32 = cv.threshold(
        img, 127, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

    # 显示图像
    fig, ax = plt.subplots(3, 3)
    ax[0, 0].set_title('THRESH_BINARY,50', fontsize=6)
    # matplotlib显示图像为rgb格式
    ax[0, 0].imshow(cv.cvtColor(img_ret11, cv.COLOR_BGR2RGB))
    ax[0, 1].set_title('THRESH_BINARY,127', fontsize=6)
    ax[0, 1].imshow(cv.cvtColor(img_ret12, cv.COLOR_BGR2RGB))
    ax[0, 2].set_title('THRESH_BINARY_INV,200', fontsize=6)
    ax[0, 2].imshow(cv.cvtColor(img_ret13, cv.COLOR_BGR2RGB))
    ax[1, 0].set_title('THRESH_TRUNC,127', fontsize=6)
    ax[1, 0].imshow(cv.cvtColor(img_ret21, cv.COLOR_BGR2RGB))
    ax[1, 1].set_title('THRESH_TOZERO,127', fontsize=6)
    ax[1, 1].imshow(cv.cvtColor(img_ret22, cv.COLOR_BGR2RGB))
    ax[1, 2].set_title('THRESH_TOZERO_INV,127', fontsize=6)
    ax[1, 2].imshow(cv.cvtColor(img_ret23, cv.COLOR_BGR2RGB))
    ax[2, 0].set_title('THRESH_OTSU', fontsize=6)
    ax[2, 0].imshow(cv.cvtColor(img_ret31, cv.COLOR_BGR2RGB))
    ax[2, 1].set_title('THRESH_TRIANGLE', fontsize=6)
    ax[2, 1].imshow(cv.cvtColor(img_ret32, cv.COLOR_BGR2RGB))
    ax[2, 2].set_title('RAW', fontsize=6)
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


def func4():
    '''自适应阈值处理(注意由于阈值并非全局而是局部阈值，范围内像素值实际很接近，因此C值实际是比较敏感的，无需设置过大)'''
    raw = cv.imread('./images/Lenna.jpg', 0)
    img = cv.medianBlur(raw, 5)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 11, 2)
    titles = ['raw', 'global', 'mean', 'Gaussian']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def func5():
    img = cv.imread('./images/3.jpg', 0)
    # 全局阈值
    ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # Otsu 阈值
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # 经过高斯滤波的 Otsu 阈值
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # 画出所有的图像和他们的直方图
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['raw', 'Histogram', 'global(v=127)',
              'raw', 'Histogram', "Otsu",
              'gaussian filtered', 'Histogram', "Otsu"]
    for i in range(3):
        plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    func5()
