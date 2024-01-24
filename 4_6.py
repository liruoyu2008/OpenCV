'''图像梯度'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def func1():
    raw = cv.imread('./images/7.png', cv.IMREAD_GRAYSCALE)
    laplacian = cv.Laplacian(raw, cv.CV_64F)
    sobelx = cv.Sobel(raw, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(raw, cv.CV_64F, 0, 1, ksize=5)
    imgs = {'raw': raw, 'laplacian': laplacian, 'sobelx': sobelx, 'sobely': sobely}
    i = 0
    for name, img in imgs.items():
        plt.subplot(3, 3, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(name, fontsize=6)
        plt.axis("off")
        i += 1
    plt.show()
    
    
def func2():
    img = cv.imread('./images/6.png',0)
    # 此处输出8u的图，部分处理细节丢失（负值梯度和溢出梯度）
    sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
    # 此处输出64F的图，可以保留更多处理细节
    sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    func1()