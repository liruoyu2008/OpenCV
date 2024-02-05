import cv2 as cv
from video_cap import VideoCap
import numpy as np


class faceDetect(VideoCap):
    # 加载预训练的 Haar Cascade 模型
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def handleImage(self, image: cv.typing.MatLike) -> cv.typing.MatLike:
        '''目标检测'''
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # 进行面部检测
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 在检测到的面部周围画矩形框
        for (x, y, w, h) in faces:
            cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return image


if __name__ == "__main__":
    fd = faceDetect()
    fd.showCap(0, cv.CAP_DSHOW)
