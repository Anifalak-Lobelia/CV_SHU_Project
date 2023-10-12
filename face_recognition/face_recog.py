import os
import cv2
import numpy as np
import face_recognition.pre_process as u
from face_recognition.network import InceptionResNetV1
from face_recognition.network import mtcnn

class face_1():
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))  # 获取当前文件的目录
        pnet_model_path = os.path.join(dir_path, 'model_data', 'pnet.h5')  # 构建pnet模型文件路径
        rnet_model_path = os.path.join(dir_path, 'model_data', 'rnet.h5')  # 构建rnet模型文件路径
        onet_model_path = os.path.join(dir_path, 'model_data', 'onet.h5')  # 构建onet模型文件路径
        self.m1 = mtcnn(pnet_model_path, rnet_model_path, onet_model_path)  # 创建MTCNN实例，用于人脸检测
        self.t1 = [0.5, 0.6, 0.8]  # 设置MTCNN的阈值
        self.f1 = InceptionResNetV1()  # 创建InceptionResNetV1实例，用于人脸识别
        model_path = os.path.join(dir_path, 'model_data', 'facenet_keras.h5')  # 加载预训练的facenet模型权重
        self.f1.load_weights(model_path)
        f_list = os.listdir(os.path.join("face_dataset"))  # 获取人脸数据集目录下的文件列表
        self.known_face_encodings = []  # 存储已知人脸编码
        self.known_face_names = []  # 存储已知人脸的名称
        for face in f_list:
            name = face.split(".")[0]  # 获取人脸图像的文件名作为人脸名称
            img = cv2.imread("./face_dataset/" + face)  # 读取人脸图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像转换为RGB格式
            rectangles = self.m1.detectFace(img, self.t1)  # 使用MTCNN检测人脸框
            rectangles = u.rect2square(np.array(rectangles))  # 将检测到的人脸框转换为正方形
            rectangle = rectangles[0]  # 取第一个人脸框
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])  # 提取关键点坐标并对齐
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]  # 根据人脸框裁剪人脸图像
            crop_img, _ = u.Alignment_1(crop_img, landmark)  # 对齐裁剪后的人脸图像
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)  # 调整图像尺寸为160x160，并扩展维度

            face_encoding = u.calc_128_vec(self.f1, crop_img)  # 计算人脸的128维编码

            self.known_face_encodings.append(face_encoding)  # 存储已知人脸编码
            self.known_face_names.append(name)  # 存储已知人脸的名称

    def recog(self, draw):
        height, width, _ = np.shape(draw)  # 获取视频帧的尺寸
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)  # 将视频帧转换为RGB格式
        rectangles = self.m1.detectFace(draw_rgb, self.t1)  # 使用MTCNN检测人脸框
        if len(rectangles) == 0:
            return
        rectangles = u.rect2square(np.array(rectangles, dtype=np.int32))  # 将检测到的人脸框转换为正方形
        rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)  # 对超出图像边界的坐标进行裁剪
        rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)  # 对超出图像边界的坐标进行裁剪
        face_encodings = []
        for rectangle in rectangles:
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])  # 提取关键点坐标并对齐
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]  # 根据人脸框裁剪人脸图像

            crop_img, _ = u.Alignment_1(crop_img, landmark)  # 对齐裁剪后的人脸图像
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)  # 调整图像尺寸为160x160，并扩展维度

            face_encoding = u.calc_128_vec(self.f1, crop_img)  # 计算人脸的128维编码
            face_encodings.append(face_encoding)  # 存储当前帧中检测到的人脸编码
        face_names = []
        for face_encoding in face_encodings:
            matches = u.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)  # 比较当前帧中的人脸编码与已知人脸编码的相似度
            name = "Unknown"  # 默认为未知人脸

            face_distances = u.face_distance(self.known_face_encodings, face_encoding)  # 计算当前帧中的人脸编码与已知人脸编码的欧氏距离

            best_match_index = np.argmin(face_distances)  # 找到欧氏距离最小的索引
            if matches[best_match_index]:  # 判断是否匹配成功
                name = self.known_face_names[best_match_index]  # 获取匹配成功的人脸的名称
            face_names.append(name)  # 存储当前帧中每个人脸的名称

        rectangles = rectangles[:, 0:4]  # 提取人脸框的坐标

        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)  # 绘制人脸框
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left, bottom - 15), font, 0.75, (255, 255, 255), 2)  # 在人脸框上方绘制人脸的名称
        return draw
    def open_camera(self):
        self.video_capture = cv2.VideoCapture(0)  # 打开摄像头

        while True:
            ret, draw = self.video_capture.read()  # 读取视频帧
            self.recog(draw)  # 人脸识别
            cv2.imshow('Video', draw)  # 显示视频帧
            if cv2.waitKey(20) & 0xFF == ord('q'):  # 按下q键退出循环
                break

        self.video_capture.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 关闭窗口


if __name__ == "__main__":
    dudu = face_1()  # 创建face_1类实例
    video_capture = cv2.VideoCapture(0)  # 打开摄像头

    while True:
        ret, draw = video_capture.read()  # 读取视频帧
        dudu.recog(draw)  # 人脸识别
        cv2.imshow('Video', draw)  # 显示视频帧
        if cv2.waitKey(20) & 0xFF == ord('q'):  # 按下q键退出循环
            break

    video_capture.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口
