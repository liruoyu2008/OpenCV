'''HOG特征+SVM支持向量机分类实现手写数字识别'''
import cv2 as cv
import numpy as np

SZ=20
bin_n = 16 # Number of bins

affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

# 使用图像的二阶矩预校正图像的偏斜
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img


# HoG特征
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # 在(0..16)范围内量化
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist是一个64维向量
    return hist

img = cv.imread('./images/digits.png',0)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")

cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# 前一半训练数据，后一半测试数据
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]

mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)    # 打印精度