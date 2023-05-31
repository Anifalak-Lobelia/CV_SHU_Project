import sys
from tensorflow import keras
import cv2

xmlfile = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
xmlfile.load('haarcascade_frontalface_alt2.xml')

# 定义情绪识别类
class emotion_recog():
    def __init__(s):
        # 加载预训练模型
        s.model = keras.models.load_model('E:/data_sets/model.h5')
        s.h=48
        s.w=48
        s.batch_size=64
        # 定义分类标签
        s.class_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        s.predict_class = []  # 初始化predict_class

    # 进行情绪识别的方法
    def emotion(s):
        try:
            # 创建ImageDataGenerator
            s.datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
            # 从文件夹中读取图片
            s.generator = s.datagen.flow_from_directory('./predict', target_size=(s.h, s.w), batch_size=s.batch_size, seed=11, shuffle=False, class_mode='categorical')
            # 进行情绪预测
            s.predict = s.model.predict(s.generator)
            print('predict_img:', s.predict)  # 打印模型预测结果
            s.predict_indices = npy.argmax(s.predict, axis=1)
            print('predict_class_indices:', s.predict_indices)  # 打印类别指数
            s.predict_class = [s.class_names[index] for index in s.predict_indices]
            print(s.predict_class)
        except Exception as e:
            print('Error in emotion method:', e)  # 打印异常信息

    # 视频识别方法
    def video_recog(s):
        s.cap = cv2.VideoCapture(0)
        while (1):
            ret, picture = s.cap.read()
            # 进行人脸及情绪识别
            s.recog(picture)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        s.cap.release()
        cv2.destroyAllWindows()

    # 人脸及情绪识别方法
    def recog(s, picture):
        grey = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        xml = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        xml.load('haarcascade_frontalface_alt2.xml')
        face = xml.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
        if len(face):
            for Rect in face:
                x, y, w, h = Rect
                # 画出人脸框
                picture=cv2.rectangle(picture, (x, y), (x + h, y + w), (0, 255, 0), 2)
                f = cv2.resize(grey[y:(y + h), x:(x + w)], (48, 48))
                cv2.imwrite('./predict/img/1.jpg', f)
                # 进行情绪识别
                emotion.emotion()
                # 将识别结果显示在画面上
                cv2.putText(picture, str(emotion.predict_class), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        # 显示画面
        cv2.imshow("Image", picture)
        # 当输入q时，退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

    # 图片情绪识别方法
    def img_recog(s, img_path):
        # 读取图片
        picture = cv2.imread(img_path, 1)
        # 进行人脸及情绪识别
        s.recog(picture)
