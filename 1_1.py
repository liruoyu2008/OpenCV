import numpy as np
import cv2 as cv

img = cv.imread('./images/1.jpg')
cv.imshow('image', img)
k = cv.waitKey(0)
if k == 27:  # 等待ESC退出
    cv.destroyAllWindows()
elif k == ord('s'):  # 等待关键字，保存和退出
    cv.imwrite('5.jpg', img)
    cv.destroyAllWindows()
