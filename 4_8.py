'''
图像金字塔
利用拉普拉斯金字塔记录图像的信息，并还原成原图像
注意还原的图像会和原图存在一定差别（类似轻微模糊效果），原因在于：
1. 插值过程中的信息丢失
2. 边缘效应
3. 数值精度问题
4. 溢出和饱和处理
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def stack():
    '''
    图像的堆叠
    无论维度，横向堆叠默认在第二个轴上连接数组，加长第二轴；
    与之对应，纵向则默认在第一轴上连接数组，加长第一轴
    '''
    A = cv.imread('./images/apple.png')
    A_rows, A_cols, A_depth = A.shape
    B = cv.imread('./images/orange.png')
    B_rows, B_cols, B_depth = B.shape
    img = np.hstack([A[:, 0:A_cols//2], B[:, B_cols//2:]])
    cv.imshow('stack', img)
    cv.waitKey(0)


def pyrDownAndRestore():
    # 1.读图，并整形，便于金字塔收缩过程中不出现奇数尺寸
    raw = cv.imread('./images/apple.png')

    # 2.创建高斯金字塔
    G = raw.copy()
    pyrs = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        pyrs.append(G)
    # 3.创建拉普拉斯金字塔
    # 拉普拉斯金字塔形象的说法，实际是源图像与其缩小放大之后的差值，相当于是失真的部分
    # 因此拉普拉斯金字塔大部分像素是黑色，故可以用来作图像压缩还原。
    # 此处以首元素gpA5为底图，剩余元素为差图，因此，从前往后按“up+后图”的方式即可还原得到原图
    laps = [pyrs[6]]
    for i in range(6, 0, -1):
        GE = cv.pyrUp(pyrs[i])
        L = cv.subtract(pyrs[i-1], GE)
        laps.append(L)
    # 4.还原
    # 最终效果未能完
    ress = [laps[0]]
    restore = ress[0]
    for i in range(0, 6):
        restore = cv.pyrUp(restore)
        restore = cv.add(restore, laps[i+1])
        ress.append(restore)
    # 5.列出系列图
    fig, plots = plt.subplots(3, 7)
    for x in range(len(pyrs)):
        plots[0, 6-x].imshow(cv.cvtColor(pyrs[x], cv.COLOR_BGR2RGB))
        plots[0, 6-x].set_title(f'pyr:{x}')
        plots[0, 6-x].axis("off")
    for y in range(len(laps)):
        plots[1, y].imshow(cv.cvtColor(laps[y], cv.COLOR_BGR2RGB))
        plots[1, y].set_title(f'lap:{6-y}')
        plots[1, y].axis("off")
    for z in range(len(ress)):
        plots[2, z].imshow(cv.cvtColor(ress[z], cv.COLOR_BGR2RGB))
        plots[2, z].set_title(f'restore:{6-z}')
        plots[2, z].axis("off")
    plt.show()


if __name__ == "__main__":
    pyrDownAndRestore()
