import cv2
import argparse
import time

def detect_and_draw_boxes(network, input_frame, threshold=0.7):
    frame_copy = input_frame.copy()
    height, width = frame_copy.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)
    network.setInput(blob)
    detected_objects = network.forward()
    bounding_boxes = []
    for i in range(detected_objects.shape[2]):
        conf = detected_objects[0, 0, i, 2]
        if conf > threshold:
            x_start = int(detected_objects[0, 0, i, 3] * width)
            y_start = int(detected_objects[0, 0, i, 4] * height)
            x_end = int(detected_objects[0, 0, i, 5] * width)
            y_end = int(detected_objects[0, 0, i, 6] * height)
            bounding_boxes.append([x_start, y_start, x_end, y_end])
            cv2.rectangle(frame_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), int(round(height/150)), 8)
    return frame_copy, bounding_boxes

parser = argparse.ArgumentParser(description='Age and Gender')
parser.add_argument('--input')
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ages = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genders = ['Male', 'Female']
faceProto, faceModel = "opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb"
ageProto, ageModel = "age_deploy.prototxt", "age_net.caffemodel"
genderProto, genderModel = "gender_deploy.prototxt", "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

video_capture = cv2.VideoCapture(args.input if args.input else 0)
padding = 20

while cv2.waitKey(1) < 0:
    start_time = time.time()
    ret, frame = video_capture.read()
    if not ret:
        cv2.waitKey()
        break
    frame_with_boxes, boxes = detect_and_draw_boxes(faceNet, frame)
    if not boxes:
        continue
    for box in boxes:
        face = frame[max(0,box[1]-padding):min(box[3]+padding,frame.shape[0]-1),max(0,box[0]-padding):min(box[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        gender_predictions = genderNet.forward()
        gender = genders[gender_predictions[0].argmax()]
        print(f"性别 : {gender}, 可信度 = {round(gender_predictions[0].max(), 3)}")
        ageNet.setInput(blob)
        age_predictions = ageNet.forward()
        age = ages[age_predictions[0].argmax()]
        print(f"年龄 : {age}, 可信度 = {round(age_predictions[0].max(), 3)}")
        info = f"{gender},{age}"
        cv2.putText(frame_with_boxes, info, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Age-Gender", frame_with_boxes)
    print(f"Time : {round(time.time() - start_time, 3)}")
