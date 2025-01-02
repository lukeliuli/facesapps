# 1.要实现从摄像头实时检测人脸的功能，我们可以使用OpenCV来捕获视频流，并使用MTCNN进行人脸检测
#2.在检测到的人脸周围绘制矩形框，并添加标号，以便区分不同的人脸
    
import cv2
from mtcnn import MTCNN

# 创建MTCNN检测器对象
detector = MTCNN()

def detect_faces_camera_with_labels():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    face_count = 0  # 初始化人脸计数器

    while True:
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # 将图像从BGR转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用MTCNN检测人脸
        faces = detector.detect_faces(frame_rgb)

        # 在原图上绘制人脸框和关键点，并添加标号
        for face in faces:
            x, y, width, height = face['box']
            face_count += 1  # 增加人脸计数器
            label = f"Face {face_count}"  # 创建标签文本
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            keypoints = face['keypoints']
            cv2.circle(frame, keypoints['left_eye'], 2, (0, 255, 0), 2)
            cv2.circle(frame, keypoints['right_eye'], 2, (0, 255, 0), 2)
            cv2.circle(frame, keypoints['nose'], 2, (0, 255, 0), 2)
            cv2.circle(frame, keypoints['mouth_left'], 2, (0, 255, 0), 2)
            cv2.circle(frame, keypoints['mouth_right'], 2, (0, 255, 0), 2)

        # 显示结果图像
        cv2.imshow('Faces Detected', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

# 测试从摄像头实时检测人脸并添加标号功能
detect_faces_camera_with_labels()
