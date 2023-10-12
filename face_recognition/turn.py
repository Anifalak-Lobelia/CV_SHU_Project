import os
import cv2
import numpy as np
import face_recognition.pre_process as utils
from face_recognition.network import InceptionResNetV1
from face_recognition.network import mtcnn

class FaceTurner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.threshold = [0.75, 0.75, 0.75]
        dir_path = os.path.dirname(os.path.realpath(__file__))  # 获取当前文件的目录
        pnet_model_path = os.path.join(dir_path, 'model_data', 'pnet.h5')  # 构建pnet模型文件路径
        rnet_model_path = os.path.join(dir_path, 'model_data', 'rnet.h5')  # 构建rnet模型文件路径
        onet_model_path = os.path.join(dir_path, 'model_data', 'onet.h5')  # 构建onet模型文件路径
        self.mtcnn_model = mtcnn(pnet_model_path, rnet_model_path, onet_model_path)  # 添加模型文件路径
        model_path = os.path.join(dir_path, 'model_data', 'facenet_keras.h5')  # 构建facenet模型文件路径
        self.facenet_model = InceptionResNetV1()
        self.facenet_model.load_weights(model_path)

    def turn_face(self):
        img = cv2.imread(self.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rectangles = self.mtcnn_model.detectFace(img, self.threshold)

        draw = img.copy()
        rectangles = utils.rect2square(np.array(rectangles))

        for rectangle in rectangles:
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

            # 将抠图后的人脸大小改变到固定尺寸
            display_img1 = cv2.resize(cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR), (400, 400))
            cv2.imshow('旋转前', display_img1)

            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            crop_img = cv2.resize(crop_img, (160, 160))
            feature1 = utils.calc_128_vec(self.facenet_model, np.expand_dims(crop_img, 0))
            print(feature1)

        final_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

        # 将旋转后的人脸大小改变到固定尺寸
        display_img2 = cv2.resize(final_img, (400, 400))
        cv2.imshow('旋转后', display_img2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return final_img
