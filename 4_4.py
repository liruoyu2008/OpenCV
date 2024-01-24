'''图像平滑。通过控制卷积核的总体权重为0或1，可以切换展示仅突出特征还是在原图像上突出特征'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def some_filter():
    img = cv.imread('./images/6.png')
    # img = plt.imread('./images/1.jpg')
    # img = cv.cvtColor(cv.imread("6.png"), cv.COLOR_BGR2RGB)

    # 均值滤波(权重和为1，因此是在图像基本保留原始样貌的情况下进行调整)
    mean_kernel = np.array((
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111],
        [0.111, 0.111, 0.111]), dtype=np.float32)
    mean_dst = cv.filter2D(img, -1, mean_kernel)

    # sobel_buttom滤波（权重之和为0，因此是抹除原始图像，仅突出图像部分特征）
    # 此sobel_buttom卷积核是凸显横向的、由暗到亮的梯度变化（向下的正梯度，因此叫b ottom）。横向不变的或由亮到暗的变化，由于其卷积结果为0或负值，在输出图像中将显示为黑点
    sobel1_kernel = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype=np.float32)
    sobel1_dst = cv.filter2D(img, -1, sobel1_kernel)

    # sobel_left滤波（凸显向左的正向梯度，因此叫left）
    sobel2_kernel = np.array((
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]), dtype=np.float32)
    sobel2_dst = cv.filter2D(img, -1, sobel2_kernel)

    # emboss浮雕滤波（检测斜(横向与纵向)向亮度变化，左上至右下的正向梯度将被凸显）
    # 看起来像左上角有光源照射而凸显了左上边缘隐藏了右下边缘，形似浮雕效果，因为得名。
    # 将中心像素置为0/1可切换查看基于特征图/原始图的输出效果
    emboss_kernel = np.array((
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]), dtype=np.float32)
    emboss_dst = cv.filter2D(img, -1, emboss_kernel)

    # outline描边（大纲、轮廓）滤波
    outline_kernel = np.array((
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]), dtype=np.float32)
    outline_dst = cv.filter2D(img, -1, outline_kernel)

    # sharpen锐化滤波，凸显图像的中心像素，拉大与周围像素的区别，因此也能突显边缘
    # 常用于处理因模糊而丢失细节的图像
    sharpen_kernel = np.array((
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]), dtype=np.float32)
    sharpen_dst = cv.filter2D(img, -1, sharpen_kernel)

    # laplacian拉普拉斯滤波
    # 拉普拉斯算子实际是针对函数各处的同性测度，是以某个图像为基础的。
    # 此处实际上是以高斯二维正态分布函数图像为基础，因此全称为aplacian of Gaussian(LoG)，
    # 其图像是一个三峰结构，正峰-负峰-正峰，且由于奇数阶导数反映斜率，偶数阶导数反映曲率的原则，
    # 正态分布在山腰两侧具有较小的正曲率，而在山顶具有较强的负曲率，因此反映到拉普拉斯函数图像上，其负峰数值上比两个正峰更大
    # 能凸显图像的二阶梯度（图像中强度发生快速变化的区域,或理解为边缘的边缘）
    # 适用于边缘检测和图像分析，特别是在需要强调图像中边缘和细节发生变化的场景中
    laplacian_kernel = np.array((
        [0, 2, 0],
        [2, -7, 2],
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

    plt.subplot(441), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('original')
    plt.subplot(442), plt.imshow(cv.cvtColor(mean_dst, cv.COLOR_BGR2RGB)), plt.title('mean')
    plt.subplot(443), plt.imshow(cv.cvtColor(sobel1_dst, cv.COLOR_BGR2RGB)), plt.title('sobel_buttom')
    plt.subplot(444), plt.imshow(cv.cvtColor(sobel2_dst, cv.COLOR_BGR2RGB)), plt.title('sobel_left')
    plt.subplot(445), plt.imshow(cv.cvtColor(emboss_dst, cv.COLOR_BGR2RGB)), plt.title('emboss')
    plt.subplot(446), plt.imshow(cv.cvtColor(outline_dst, cv.COLOR_BGR2RGB)), plt.title('outline')
    plt.subplot(447), plt.imshow(cv.cvtColor(sharpen_dst, cv.COLOR_BGR2RGB)), plt.title('sharpen')
    plt.subplot(448), plt.imshow(cv.cvtColor(laplacian_dst, cv.COLOR_BGR2RGB)), plt.title('laplacian')
    plt.subplot(449), plt.imshow(cv.cvtColor(gaussian_dst, cv.COLOR_BGR2RGB)), plt.title('gaussian_blur')
    plt.subplot(4, 4, 10), plt.imshow(cv.cvtColor(bilateral_dst, cv.COLOR_BGR2RGB)), plt.title('bilateral')
    plt.show()


if __name__ == '__main__':
    some_filter()
