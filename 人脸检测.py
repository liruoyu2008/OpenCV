import cv2 as cv

# 加载预训练的 Haar Cascade 模型
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
while (cap.isOpened()):
    # 读取图像
    ret, image = cap.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 进行面部检测
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 在检测到的面部周围画矩形框
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示图像
    cv.imshow('Face detection', image)
    if cv.waitKey(40) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
