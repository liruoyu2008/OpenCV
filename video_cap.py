import cv2 as cv
import numpy as np


class VideoCap():
    def showCap(self, cap_index: int, cap_mode: int):
        '''开启摄像头拍摄'''
        cap = cv.VideoCapture(cap_index, cap_mode)
        while (cap.isOpened()):

            # 读取图像
            ret, image = cap.read()

            # 处理图像
            image_new = self.handleImage(image)

            # 显示图像
            cv.imshow('capture', image_new)
            if cv.waitKey(40) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    def handleImage(self, image: cv.typing.MatLike) -> cv.typing.MatLike:
        '''处理图像'''
        return image


if __name__ == "__main__":
    vc = VideoCap()
    vc.showCap(0, cv.CAP_DSHOW)
