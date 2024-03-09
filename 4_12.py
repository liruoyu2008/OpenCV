'''模板匹配（没有尺度不变性，对旋转和缩放太敏感）'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('./images/15.png', 0)
img2 = img.copy()
template = cv.imread('./images/messi_face.png', 0)
w, h = template.shape[::-1]
# 使用下列6个模板匹配函数方法分别进行匹配
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # 应用模板匹配
    res = cv.matchTemplate(img, template, method)
    # 寻找唯一的最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # 有些方法匹配度越高，得到的返回值越小，例如：TM_SQDIFF、TM_SQDIFF_NORMED
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

# 若可能存在多个匹配位置时，可使用阈值进行区分
img_rgb = cv.imread('./images/17.png')
img_rgb2 = img_rgb.copy()
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('./images/coin.png', 0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb2, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
plt.subplot(131), plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title('raw'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(res, cmap='gray')
plt.title('match result'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(cv.cvtColor(img_rgb2, cv.COLOR_BGR2RGB))
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()
