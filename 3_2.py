import cv2 as cv
import matplotlib.pyplot as plt

# 加载两张图片
img1 = cv.imread('1.jpg')
img2 = cv.imread('2.jpg')
img3 = img1[0:900, -1000:, :]
img4 = img2[0:900, -1000:, :]
i = 0
j = 1
while True:
    e1 = cv.getTickCount()
    img5 = cv.addWeighted(img3, i, img4, j, 0)
    cv.imshow('res', img5)
    cv.waitKey(1000)
    if i >= 0.99:
        break
    i += 0.1
    j -= 0.1
    e2 = cv.getTickCount()
    print((e2-e1)/cv.getTickFrequency())
cv.destroyAllWindows()
