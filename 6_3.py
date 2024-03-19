'''背景减法'''
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)


def func1():
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

    while (1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv.imshow('frame', fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def func2():
    # 灰色表示阴影
    fgbg = cv.createBackgroundSubtractorMOG2()

    while (1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv.imshow('frame', fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def func3():
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

    while (1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

        cv.imshow('frame', fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    func2()
