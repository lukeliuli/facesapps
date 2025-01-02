import cv2
import dlib

# 初始化dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break
    
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = detector(gray)
    
    # 在检测到的人脸周围绘制矩形框并添加标号
    for idx, face in enumerate(faces):
        # 获取人脸矩形框的坐标
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        
        # 绘制矩形框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标号
        cv2.putText(frame, f"Face #{idx+1}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Face Detection', frame)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()