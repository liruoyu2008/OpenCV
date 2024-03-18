'''Meanshift均值漂移聚类算法实现的目标追踪'''
import numpy as np
import cv2 as cv
from cv2.typing import MatLike

# 初始化鼠标事件的回调函数所需的参数
drawing = False                 # 当鼠标被按下时变为True
x1, y1, w1, h1 = -1, -1, 0, 0         # 当前绘制过程中的矩形
x2, y2, w2, h2 = 0, 0, 0, 0     # 绘制完毕的矩形

# 初始化Meanshift算法的参数
track_window = None
roi_hist = np.zeros((0, 0))


# 鼠标的回调函数
def draw_rectangle(event, x, y, flags, param):
    global drawing, x1, y1, w1, h1, x2, y2, w2, h2
    # 当按下左键时返回起始位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    # 当左键按下并移动时绘制矩形。event可以查看移动，flag查看是否按下
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        if drawing == True:
            x1, y1, w1, h1 = (x1, y1, x-x1, y-y1)

    # 当鼠标松开停止绘制矩形
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # 保存确定矩形框的位置和尺寸
        if x != x1 and y != y1:
            x2, y2, w2, h2 = (x1, y1, x-x1, y-y1)
            initial_roi(param, x2, y2, w2, h2)


# 初始化追踪窗口
def initial_roi(frame, x, y, w, h):
    global track_window, roi_hist
    # 设置窗口的初始位置
    track_window = (x2, y2, w2, h2)

    # 设置 ROI(图像范围)以进行跟踪
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)


# 主程序启动
# 获取视频的第一帧
cv.namedWindow('frame')
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
ret, frame = cap.read()

# 定义鼠标事件
cv.setMouseCallback('frame', draw_rectangle, frame)

# 设置结束标志，10 次迭代或至少 1 次移动
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, frame = cap.read()
    if ret == True:
        # 绘制鼠标框选的矩形区域(黄色)
        if drawing == True:
            frame = cv.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 255), 2)

        # 绘制追踪区域（红色）
        if track_window != None:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 运行 meanshift 算法用以获取新的位置
            ret, track_window = cv.meanShift(dst, track_window, term_crit)
            
            # 绘制矩形
            x, y, w, h = track_window
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # 呈现图像
        cv.imshow('frame', frame)

        # 按q结束程序
        k = cv.waitKey(60) & 0xff
        if k == ord('q'):
            break
    else:
        break

cv.destroyAllWindows()
cap.release()
