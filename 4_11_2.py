'''图像转换（傅里叶）'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# opencv中的离散傅里叶变换
# 与numpy的区别在于cv得到的是shape为[m,n,2]的实矩阵，最后一轴的元素分别表示实部和虚部
# numpy的返回则是shape为[m,n]的复矩阵
img = cv.imread('./images/15.png',0)
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT) # 离散傅里叶
dft_shift = np.fft.fftshift(dft)    # 低频中置
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) # 计算幅值
plt.subplot(331),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
rows, cols = img.shape
crow,ccol = rows//2 , cols//2
# 创建低频蒙版，使图像仅保留低频部分
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(333),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])
# 创建高频蒙版，使图像仅保留高频部分
mask2 = np.ones((rows,cols,2),np.uint8)
mask2[crow-30:crow+30, ccol-30:ccol+30] = 0
fshift = dft_shift*mask2
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(334),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.show()

# 此外，对于傅里叶变换，图像尺寸若是2的幂次，计算性能会显著增加，
# 因此，可以找到一个刚好大于原图像且尺寸为2的幂次的0图像，将原图像放置于其中进行转换和频域操作，
# 然后通过逆转换和图像裁切重新得到原来尺寸的空域图
# 获得最佳尺寸的函数：cv.getOptimalDFTSize()
# 将原图扩大为该尺寸的函数： cv.copyMakeBorder()