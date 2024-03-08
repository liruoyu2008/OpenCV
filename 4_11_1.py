'''图像转换（傅里叶）'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# numpy中的离散傅里叶变换
img = cv.imread('./images/15.png',0)
plt.subplot(331),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# 傅里叶转换
f = np.fft.fft2(img)
# 低频中置
fshift = np.fft.fftshift(f)
# 对复矩阵求模（幅度）得到幅度频谱图
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(332),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# 用60*60的中心遮罩滤除频域的低频成分（等于做了一次高通滤波HPF）
rows, cols = img.shape
crow,ccol = rows//2 , cols//2
# ！！这里直接赋0值会使原本预计要变黑的中心频谱直接变全白，导致频谱图和最终结果图出现错误，
# 原因是频谱图计算幅值并取对数时，真数为0会返回负无穷。。。可以用一个非常小的数据替代0即可解决
fshift[crow-30:crow+31, ccol-30:ccol+31] = 0.01
magnitude_spectrum2 = 20*np.log(np.abs(fshift))
plt.subplot(333),plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Magnitude Spectrum2'), plt.xticks([]), plt.yticks([])
# 逆低频中置
f_ishift = np.fft.ifftshift(fshift)
# 逆傅里叶变换（得到的实矩阵仅取其实部即可，理论上虚部均为0）
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)
plt.subplot(334),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.show()