'''相机校准'''
'''棋盘格图案的角点具有很高的对比度，并且分布规则，使得它们可以被自动检测并用于精确地估计相机参数。
OpenCV中的cv.findChessboardCorners函数就是专门设计来自动寻找这些角点的，这对于相机校正来说是一种
非常高效且可靠的方法。
棋盘格提供了一组已知的、规则排列的二维点，这些点在三维空间中的布局是固定的。通过分析棋盘格在不同拍摄角度
下的图像，可以计算出相机镜头的畸变参数以及相机相对于这些已知点的位置和旋转角度。这样就可以进一步使用这些
信息来校正其他图像中的畸变，改善图像质量，或者用于三维场景的重建和测量等应用。棋盘格之所以被广泛使用，正是
因为它的简单性、高效性以及在计算机视觉任务中的高度适用性。'''

# 终止标准
import numpy as np
import cv2 as cv
import glob
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 标定
# 假定以某个角落的内角点为原点选定坐标系，单元格长度为单位1，
# 则这些标定的内角点为 (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0),并将其网格化赋予到预设的3D坐标
# 根据后面cv.drawChessboardCorners画出的点来看，cv.calibrateCamera会认为我的坐标系是以右下角的
# 内角点为原点，x轴向左、y轴向上。
# 因此，自己选定的坐标系的原点位置、轴方向不重要、只要在选定坐标系后，保持objpoints与imgpoints
# 一一对应，没有局部错乱即可。
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
camera_size = (-1, -1)

# 用于存储所有图像对象点与图像点的矩阵
objpoints = []  # 在真实世界中的 3d 点
imgpoints = []  # 在图像平面中的 2d 点

# 读取所有的符合规则的文件路径(准备好相机拍摄的一批棋盘格图案)
# 使用该待估计的相机拍摄的多张不同视角和位置的棋盘格样张是必须的，一张或少量的图像会导致参数
# 估计不足（例如：相机深度信息）
# 理想的相机校正过程通常需要从不同的角度和位置拍摄多张棋盘格图像。具体需要多少张图像，取决于校正过程的
# 要求以及所用方法的具体细节，但是一般建议至少拍摄 10 张以上的图像。这些图像应该覆盖相机视野的不同部分，
# 包括中心、边缘、角落等区域，并且棋盘格应该在每张图像中以不同的角度出现，包括倾斜和旋转。
images = glob.glob('./images/chessboard-*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if camera_size == (-1, -1):
        camera_size = gray.shape

    # 找到棋盘上所有的角点
    # 如果棋盘在相机中未完整出现的话，patternSize应传入在相机中露出的内角点布局。
    # 不过，虽说只拍摄部分棋盘格是完全可行的，但这会为预期的patternSize和实际检测到的布局之间带来较大的分歧。
    # 主要是因为部分棋盘格子被遮挡的话，会影响函数对它周围的内角点的检测。
    # 为了避免这个不必要的分歧，实际操作中需要在所有拍摄的棋盘格图像中都露出完整的棋盘。
    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)

    # 如果找到了，便添加对象点和图像点(在细化后)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # 绘制角点
        cv.drawChessboardCorners(img, (7, 7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)

# 校准
# 使用所有得到的点的真实三维结构（objpoints）与图像的二维结构（imgpoints）映射进行参数估计
# 入参
# objectPoints: 世界坐标系中的点的三维坐标。通常是一个列表，其中每个元素都是一个表示单个视图中所有标定板角点的numpy.ndarray。
# imagePoints: 图像坐标系中点的二维坐标。它的布局应与objectPoints相匹配。
# imageSize: 用于标定的图像的大小，格式为(width, height)。
# cameraMatrix: 相机矩阵或内参矩阵A。如果使用flags参数指定了CALIB_USE_INTRINSIC_GUESS，则作为初始估计输入；否则，它仅作为输出，由函数填充。
# distCoeffs: 畸变系数，用于描述和校正相机镜头的畸变。这同样是一个输出参数，由函数填充。
# flags: 方法的标志，例如CALIB_FIX_PRINCIPAL_POINT将固定主点在中心，CALIB_ZERO_TANGENT_DIST将假设没有切向畸变等。
# criteria: 迭代算法的终止准则。它可以指定最大迭代次数和/或精度。
# 返回值
# retval: 重投影误差（单位：像素），表示估计的参数准确性的一个度量，越小表明估计越准确。
# cameraMatrix: 内参矩阵，包含焦距(fx, fy)和主点(cx, cy)的信息。
# distCoeffs: 畸变系数，包括径向畸变和切向畸变参数。
# rvecs: 每个视图的旋转向量，表示对象相对于相机的旋转。
# tvecs: 每个视图的平移向量，表示对象相对于相机的平移。
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, camera_size, None, None)

# 矫正
img = cv.imread('./images/chessboard-01.jpg')
h,  w = img.shape[:2]
# 原mtx反映了相机的实际内参，但是依据此内参进行图像校正的话，得到的矫正后图像一般还需要经过适当调整(存在黑边之类）。
# 而newCameraMatrix就是直接通过微调调整内参将“调整矫正后图像”这一步骤固化下来，后续通过newCameraMatrix进行矫正的
# 图像就免除了这一步骤，直接得到能用的目标图像。
# roi是掩膜，用于框定原始图像经过矫正后的新的像素（可能都不是矩形区域了）
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 直接矫正
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# 或者，找到映射函数，然后重映射
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# 使用roi裁切图像
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult', dst)
cv.waitKey(0)

# 重投影误差
# 利用估计的参数将真实点投影为平面点（重投影），计算新的平面点与原始平面点的匹配程度。
# 此处用L2范数（即均方误差）进行衡量
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))

cv.destroyAllWindows()
