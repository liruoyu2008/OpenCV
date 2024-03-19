'''Farneback稠密光流算法'''
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while (1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # prev: 输入图像的前一帧，必须为单通道的浮点型图像。
    # next: 输入图像的当前帧，与前一帧大小和类型相同。
    # flow: 计算出的光流场，与原图像有相同的大小，但类型为CV_32FC2。这个参数可以设为None，函数会自动计算光流。
    # pyr_scale: 图像金字塔之间的缩放系数。小于 1 的值，例如 0.5，表示每一层图像尺寸都是前一层的一半，这有助于检测在原始图像中无法检测到的快速运动。
    # levels: 图像金字塔的层数，包含原始图像层。
    # winsize: 平均窗口大小，用于计算多项式展开的参数。值越大，算法能够检测到更快的运动，但需要更多的计算时间。
    # iterations: 每一层金字塔进行迭代的次数。
    # poly_n: 用于多项式展开的邻域大小。通常取值为 5 或 7。
    # poly_sigma: 标准差，用于高斯权重。通常与 poly_n 一起使用。
    # flags: 操作标志，如计算方法或插值方法等。
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 将得到的x、y方向上的位移转化到极坐标系，得到每个像素点两帧运动间发生的角度差和极半径差
    # 角度差总值360度（这里的ang实际是弧度单位，需要乘以180/pi转化一下）的一半刚好吻合色相值，
    # 因此使用对应值为原像素的色相赋值以通过颜色变化直观的查看像素的运动角度。
    # 极半径差值则被归一化到0~255范围内，同时赋值给原像素的明度，根据明度直观的查看运行的幅度
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

cap.release()
cv.destroyAllWindows()
