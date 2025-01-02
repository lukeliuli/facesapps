from ultralytics import YOLO
import cv2
import time
import os
import wget
from pathlib import Path

def download_model():
    model_path = Path('models/yolov8n-face.pt')
    if not model_path.exists():
        print("Downloading YOLOv8 face detection model...")
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True)
        url = "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
        wget.download(url, str(model_path))
        print("\nModel downloaded successfully!")
    return str(model_path)

def detect_faces_realtime():
    # 确保模型文件存在
    #model_path = download_model()
    
    # 加载YOLOv8人脸检测模型
    model = YOLO('.\yolov8n-face.pt')
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 用于计算FPS
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用YOLOv8进行人脸检测
        results = model(frame, conf=0.5)
        
        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # 在图像上绘制检测结果
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 获取置信度
                conf = float(box.conf[0])
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label = f"Face #{i+1} ({conf:.2f})"
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(frame, label, (x1, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示检测到的人脸数量
        if len(results) > 0:
            face_count = len(results[0].boxes)
            cv2.putText(frame, f"Faces: {face_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('YOLOv8 Face Detection', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detect_faces_realtime()
    except Exception as e:
        print(f"Error: {e}")