import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
import cv2
import mediapipe as mp
import numpy as np

import torch
from torchvision import transforms, models
from PIL import Image

import torch.nn as nn

def main():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    model = models.mobilenet_v3_small()
    num_ftrs = model.classifier[3].in_features  # the Linear layer is at index 1
    model.classifier[3] = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load("mobilenet_mnist_best.pth"))
    
    # 检查CUDA是否可用
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if cuda_available:
        print("CUDA可用，使用GPU加速进行图像缩放")
    else:
        print("CUDA不可用，使用CPU处理")
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 创建窗口
    cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("无法捕获视频帧")
            break
        # 获取原始帧的尺寸
        (h, w) = frame.shape[:2]
        # 根据原始长宽比计算调整后的高，假设调整宽度为1280
        new_width = 1080
        new_height = int(new_width * h / w)
        frame = cv2.resize(frame, (new_width, new_height))
          # 在显示前将帧左右镜像
        frame = cv2.flip(frame, 1)

        if not hasattr(main, 'hands'):
            main.mp_hands = mp.solutions.hands
            main.hands = main.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            main.mp_drawing = mp.solutions.drawing_utils

        def is_fist(hand_landmarks, width, height):
            # 根据手腕与食指指尖之间的距离与手部大小判断是否为握拳
            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]
            wrist_pos = (wrist.x * width, wrist.y * height)
            index_tip_pos = (index_tip.x * width, index_tip.y * height)
            dx = index_tip_pos[0] - wrist_pos[0]
            dy = index_tip_pos[1] - wrist_pos[1]
            dist = (dx**2 + dy**2) ** 0.5
            # 计算手部边界框尺寸作为归一化的依据
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            box_width = (max(x_coords) - min(x_coords)) * width
            box_height = (max(y_coords) - min(y_coords)) * height
            hand_size = max(box_width, box_height)
            # 如果食指指尖与手腕的距离小于手部尺寸的一定比例，则认为手为握拳
            return dist < hand_size * 0.9

        if not hasattr(main, 'current_trajectory'):
            main.current_trajectory = []
            canvas = 255 * np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            num = 999

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = main.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                main.mp_drawing.draw_landmarks(frame, hand_landmarks, main.mp_hands.HAND_CONNECTIONS)
                if is_fist(hand_landmarks, w, h):
                    if main.current_trajectory == []:
                        continue
                    else:
                        main.current_trajectory = []
                        # 统计画布中被绘制的部分面积，仅当面积超过阈值时保存轨迹并调用分类函数
                        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                        drawn_area = cv2.countNonZero(mask)
                        threshold_area = 3000  # 可根据实际情况调整阈值
                        if drawn_area > threshold_area:
                            cv2.imwrite("trajectory.png", canvas)
                            num = minist(transform=transform, model=model)
                        canvas = 255 * np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                        # return
                else:
                    index_finger_tip = hand_landmarks.landmark[8]
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    cv2.circle(frame, (cx, cy), 16, (0, 0, 255), -1)
                    cv2.putText(frame, f"({cx},{cy})", (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    main.current_trajectory.append((cx, cy))
        else:
            main.current_trajectory = []
            
        # 正常绘制当前轨迹
        for i in range(1, len(main.current_trajectory)):
            cv2.line(frame, main.current_trajectory[i - 1], main.current_trajectory[i], (193,182,255), 18)
            cv2.line(canvas, main.current_trajectory[i - 1], main.current_trajectory[i], (255, 255, 255), 18)
        
        cv2.putText(frame, f"predicted: {num}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
        cv2.imshow("Camera", frame)
        
        # 按空格键退出
        if cv2.waitKey(30) & 0xFF == 32:
            break

    # 释放摄像头资源并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

def minist(transform=None,model=None):
    # 加载保存的图像文件 (确保文件路径正确)
    img_path = "trajectory.png"
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("无法加载图像:", e)
        exit(1)

    # 应用预定义的转换并增加一个批处理维度
    input_tensor = transform(image).unsqueeze(0)

    # 设置模型为评估模式并执行推理
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    print(outputs)
    #print("预测的类别为:", predicted.item())
    return  predicted.item()


if __name__ == '__main__':
    main()