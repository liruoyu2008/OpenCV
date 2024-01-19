'''图像平滑'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def some_filter():
    img = plt.imread('./images/1.jpg')
    # img = cv.cvtColor(cv.imread("1.jpg"), cv.COLOR_BGR2RGB)

    # 均值滤波(权重和为1，因此是基于图像进行微调)
    mean_kernel = np.array((
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111]), dtype=np.float32)
    mean_dst = cv.filter2D(img, -1, mean_kernel)

    # sobel_buttom滤波（凸显横向梯度变化。权重之和为0，因此是为了展示图像变化而抹除不变）
    sobel1_kernel = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype=np.float32)
    sobel1_dst = cv.filter2D(img, -1, sobel1_kernel)

    # sobel_left滤波（凸显垂直梯度变化）
    sobel2_kernel = np.array((
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]), dtype=np.float32)
    sobel2_dst = cv.filter2D(img, -1, sobel2_kernel)

    # emboss浮雕滤波（凸显横、纵、右斜梯度变化）
    emboss_kernel = np.array((
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]), dtype=np.float32)
    emboss_dst = cv.filter2D(img, -1, emboss_kernel)

    # outline描边（大纲、轮廓）滤波
    outline_kernel = np.array((
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]), dtype=np.float32)
    outline_dst = cv.filter2D(img, -1, outline_kernel)

    # sharpen锐化滤波
    sharpen_kernel = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype=np.float32)
    sharpen_dst = cv.filter2D(img, -1, sharpen_kernel)

    # laplacian拉普拉斯滤波
    laplacian_kernel = np.array((
        [0, 2, 0],
        [2, -8, 2],
        [0, 2, 0]), dtype=np.float32)
    laplacian_dst = cv.filter2D(img, -1, laplacian_kernel)

    # identity分身滤波（克隆）
    identity_kernel = np.array((
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]), dtype=np.float32)
    identity_dst = cv.filter2D(img, -1, identity_kernel)

    # gaussian_blur高斯模糊
    # gaussian_dst = cv.GaussianBlur(img, (5, 5), 0)
    gaussian_blur_kernel = cv.getGaussianKernel(19, 0)
    gaussian_dst = cv.filter2D(img, -1, gaussian_blur_kernel)

    # bilateral双边滤波
    bilateral_dst = cv.bilateralFilter(img, 9, 75, 75)

    plt.subplot(341), plt.imshow(img), plt.title('original')
    plt.subplot(342), plt.imshow(mean_dst), plt.title('mean')
    plt.subplot(343), plt.imshow(sobel1_dst), plt.title('sobel_buttom')
    plt.subplot(344), plt.imshow(sobel2_dst), plt.title('sobel_left')
    plt.subplot(345), plt.imshow(emboss_dst), plt.title('emboss')
    plt.subplot(346), plt.imshow(outline_dst), plt.title('outline')
    plt.subplot(346), plt.imshow(sharpen_dst), plt.title('sharpen')
    plt.subplot(347), plt.imshow(laplacian_dst), plt.title('laplacian')
    # plt.subplot(348), plt.imshow(identity_dst), plt.title('identity')
    plt.subplot(348), plt.imshow(gaussian_dst), plt.title('gaussian_blur')
    plt.subplot(349), plt.imshow(bilateral_dst), plt.title('bilateral')
    plt.show()


if __name__ == '__main__':
    some_filter()
