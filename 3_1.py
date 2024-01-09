import numpy as np
import cv2 as cv


def func1():
    img = cv.imread("1.jpg")
    b, g, r = cv.split(img)
    b[:] = 0
    g[:] = 0
    img2 = cv.merge((b, g, r))
    cv.imshow("b", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def func2():
    from matplotlib import pyplot as plt
    BLUE = [0, 255, 0]
    img1 = cv.imread('1.jpg')
    replicate = cv.copyMakeBorder(img1, 100, 100, 100, 100, cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(img1, 100, 100, 100, 100, cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(img1, 100, 100, 100, 100, cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(img1, 100, 100, 100, 100, cv.BORDER_WRAP)
    constant = cv.copyMakeBorder(img1, 100, 100, 100, 100, cv.BORDER_CONSTANT, value=BLUE)
    plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
    plt.show()


if __name__=="__main__":
    func1()