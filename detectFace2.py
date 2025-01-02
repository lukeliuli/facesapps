import torch
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import wget
from pathlib import Path

from FaceBoxesPyTorch.models.faceboxes import FaceBoxes    

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def download_weights():
    weights_path = Path('FaceBoxes.pth')
    if not weights_path.exists():
        print("Downloading FaceBoxes weights...")
        if not weights_path.parent.exists():
            weights_path.parent.mkdir(parents=True)
        url = "https://github.com/zisianw/FaceBoxes.PyTorch/releases/download/v1.0/FaceBoxes.pth"
        wget.download(url, str(weights_path))
        print("\nWeights downloaded successfully!")
    return str(weights_path)


    def __init__(self, phase='test', size=1024):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.size = size
        
        # CONV layers
        self.conv1 = BasicConv2d(3, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = BasicConv2d(48, 64, kernel_size=5, stride=2, padding=2)
        
        # Inception layers
        self.inception1 = nn.Sequential(
            BasicConv2d(64, 32, kernel_size=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1)
        )
        self.inception2 = nn.Sequential(
            BasicConv2d(64, 32, kernel_size=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1)
        )
        




        # Detection layers
        self.det_conv1 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
        self.det_conv2 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        det1 = self.det_conv1(x)
        det2 = self.det_conv2(x)
        return det1, det2

def nms(boxes, scores, threshold=0.5):
    """非极大值抑制"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    _, order = scores.sort(0, descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    
    return torch.LongTensor(keep)

def process_detections(det1, det2, conf_threshold=0.5):
    """处理检测结果"""
    # 将检测结果转换为边界框和分数
    boxes1 = det1[:, :4].reshape(-1, 4)
    scores1 = det1[:, 4]
    boxes2 = det2[:, :4].reshape(-1, 4)
    scores2 = det2[:, 4]
    
    # 合并两个检测层的结果
    boxes = torch.cat([boxes1, boxes2], dim=0)
    scores = torch.cat([scores1, scores2], dim=0)
    
    # 应用置信度阈值
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    # 应用NMS
    keep = nms(boxes, scores)
    boxes = boxes[keep]
    scores = scores[keep]
    
    # 转换为列表格式
    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        detections.append([x1, y1, x2, y2, score.item()])
    
    return detections

def detect_faces_realtime():
    # 下载权重
    weights_path = download_weights()
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceBoxes(phase='test',size=None,num_classes = 2).to(device)
   
    
    # 加载预训练权重

    model = load_model(model, 'FaceBoxes.pth', True)

    #model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 用于计算FPS
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 预处理图像
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
        image = image.to(device)
        
        # 人脸检测
        with torch.no_grad():
            det1, det2 = model(image)
            detections = process_detections(det1, det2)
        
        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # 绘制检测结果
        for i, det in enumerate(detections):
            x1, y1, x2, y2, score = det
            
            # 转换坐标到原始图像尺寸
            h, w = frame.shape[:2]
            x1 = int(x1 * w / 1024)
            y1 = int(y1 * h / 1024)
            x2 = int(x2 * w / 1024)
            y2 = int(y2 * h / 1024)
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            label = f"Face #{i+1} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示人脸数量
        cv2.putText(frame, f"Faces: {len(detections)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('FaceBoxes Detection', frame)
        
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