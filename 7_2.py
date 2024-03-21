'''姿势估计（已经可以根据相机姿势在空间中进行绘制了，类似于AR增强现实）'''
import numpy as np
import cv2 as cv

# 读取事先存好的数据
with np.load('./data/camera_home.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


# 根据给定的axis单元点绘制空间坐标轴
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, np.int0(corner), np.int0(tuple(imgpts[0].ravel())), (255, 0, 0), 5)
    img = cv.line(img, np.int0(corner), np.int0(tuple(imgpts[1].ravel())), (0, 255, 0), 5)
    img = cv.line(img, np.int0(corner), np.int0(tuple(imgpts[2].ravel())), (0, 0, 255), 5)
    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

# 实时读取
cv.namedWindow('img')
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
while (1):
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret2, corners = cv.findChessboardCorners(gray, (7, 7), None)
    if ret2 == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 求解多点透视（Perspective-n-Points）找到旋转和平移向量
        # 实际上calibrateCamera相机校准也能顺便求出旋转和平移向量。这两个算法的侧重点不同。
        # cali侧重点是为了求解mtx和dist，slovePnP是在相机参数已知的情况下求解空间上的变换（旋转和平移）
        ret2, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # 投射 3D 点到平面图像上
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        cv.drawChessboardCorners(img, (7, 7), corners2, ret)
        img = draw(img, corners2, imgpts)
        
    # 展示图像    
    cv.imshow('img', img)
    # 按 q 退出
    k = cv.waitKey(30) & 0xFF
    if k == ord('q'):
        break

cv.destroyAllWindows()
