import cv2
import numpy as np
import time
from policy import OpenDoBot
from PIL import Image 

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
import torchvision.transforms as T
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "logs\DSP_policy_epoch_10.ckpt" 
FRAMEWORK = "pytorch"

IMAGE_WIDTH = 256 
IMAGE_HEIGHT = 256 

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

aug = T.RandomResizedCrop((256,256), scale=(1.0, 1.0), ratio=(1.0, 1.0))

def load_model(model_path, framework):
    model = None
    if framework == "pytorch":
        model = OpenDoBot()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    return model

def predict(model, image, qpos, framework):
    action = None
    qpos = np.array(qpos, dtype=np.float32).reshape(1, -1) 
    qpos = torch.from_numpy(qpos)
    action = model(qpos,image)
    return action

# --- 主程序 ---
def main():
    model = load_model(MODEL_PATH, FRAMEWORK)
    cap = cv2.VideoCapture(0) 

    print("\n摄像头已启动。按 'q' 键退出。")
    print("在每次循环中，您将被提示输入4个qpos值。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法从摄像头读取帧。")
                break

            cv2.imshow("原始图像 (按 's' 键进行推理, 'q' 键退出)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("正在退出...")
                break
            elif key == ord('s'): 
                while True:
                    try:
                        qpos_str = input("请输入4个qpos值 (用逗号分隔, 例如: 0.1,0.2,0.3,0.4): ")
                        qpos_list = [float(x.strip()) for x in qpos_str.split(',')]
                        if len(qpos_list) == 4:
                            qpos = np.array(qpos_list)
                            break
                        else:
                            print("请输入恰好4个值。")
                    except ValueError:
                        print("输入无效，请输入用逗号分隔的数字。")
                    except Exception as e:
                        print(f"发生错误: {e}")

                print(f"获取到的 qpos: {qpos}")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_rgb)

                frame = aug(frame)
                frame = transform(frame)

                action = predict(model, frame, qpos, FRAMEWORK)

                if action is not None:
                    print(f"  模型推理输出 action: {action}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放。")

if __name__ == "__main__":
    MODEL_PATH = "logs\DSP_policy_epoch_10.ckpt" 

    FRAMEWORK = "pytorch"

    IMAGE_WIDTH = 256 
    IMAGE_HEIGHT = 256 

    main()