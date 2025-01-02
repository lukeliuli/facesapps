# 所有代码都是AI生成，其中cursor的效果最好

# detectFace1.py
## 1.要实现从摄像头实时检测人脸的功能，我们可以使用OpenCV来捕获视频流，并使用MTCNN进行人脸检测
## 2.在检测到的人脸周围绘制矩形框，并添加标号，以便区分不同的人脸

# detectface2.py(没有实现)

## 1.要实现从摄像头实时检测人脸的功能，我们可以使用OpenCV来捕获视频流，并使用FaceBoxes进行人脸检测
## 2.在检测到的人脸周围绘制矩形框，并添加标号，以便区分不同的人脸
## 3.用cusrsor生成代码比其他方式好太多，但是还有问题，不准备解决了


# detectface3.py

## 1. 要实现从摄像头实时检测人脸的功能，我们可以使用OpenCV来捕获视频流，并使用Dlib模型进行人脸检测
## 2. 在检测到的人脸周围绘制矩形框，并添加标号，以便区分不同的人脸
## 3. 创建虚拟环境   python -m venv dlibface,并激活
## 4. 百度安装各种环境，其中关键下载离线版本dlib,已经下载到win64版本的dlib文件夹中

# detectface4.py（没有完成）

## 基于centerface,但是opencv无法加载onnx模型文件
## 使用python3.6 conda create -n centerfacepy36 python=3.6
##  pip intall opencv-python==4.1.0.25,pip install scikit-learn
## onnx_importer.cpp:57: error: (-210:Unsupported format or combination of formats) Failed to parse onnx mode


# detectface5.py

## 1. 要实现从摄像头实时检测人脸的功能，我们可以使用OpenCV来捕获视频流，并使用yolov8模型进行人脸检测
## 2. 在检测到的人脸周围绘制矩形框，并添加标号，以便区分不同的人脸
## 3. pip install ultralytics 并下载yolov8n-face.pt文件
## 4. 基于yolov8是现阶段效果最好的 

# 最终选择 yolo8n




# 参考1
+ 人脸检测的预训练模型在计算机视觉领域扮演着重要角色，它们能够在各种场景下高效地检测出人脸。以下是一些常见的人脸检测预训练模型及其特点：

+ 人脸检测的预训练模型
1. OpenVINO模型库：包含MobileNetv2、SqueezeNet、ResNet152等，支持不同场景与分辨率的人脸检测，以及SSD、FCOS、ATSS等检测头。
2. RetinaFace：一个轻量级的人脸检测器，使用自定义的轻量级backbone网络(blite)，在Wider Face数据集上达到了较高的平均精度。
3. YOLO系列：如YOLO-Tiny和YOLOv3，YOLO系列模型以其高准确率和速度著称，但YOLO-Tiny在计算资源需求上有所妥协。
4. SSD：Single Shot MultiBox Detector，提供良好的准确性和推理速度，但在某些情况下可能不如YOLO系列模型。
5. BlazeFace：由谷歌开发，速度快，专为手机摄像头优化，但在处理CCTV等图像时表现不佳。
6. Faceboxes：推理速度快，准确度可与YOLO相媲美，适用于CPU上的实时检测。

# 参考2
1. https://zhuanlan.zhihu.com/p/32702868
2. https://blog.csdn.net/yuanlulu/article/details/89739643

