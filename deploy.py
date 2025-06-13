import cv2
import numpy as np
import time
from policy import OpenDoBot
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
import lib.magician.DobotDllType as dType
# -0.4019,149.6464,12.2097,90.1539

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "logs\depth\DSP_policy_epoch_200.ckpt"
FRAMEWORK = "pytorch"

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
aug = T.RandomResizedCrop((256, 256), scale=(1.0, 1.0), ratio=(1.0, 1.0))

# 标准化信息
pose_stats = {
    'X': {'mean': -10.81992619047619, 'std': 38.21256246008921},
    'Y': {'mean': 208.8902984126984, 'std': 39.440654665607966},
    'Z': {'mean': -5.283230158730158, 'std': 27.04021482166495},
    'R': {'mean': 92.70010793650793, 'std': 9.232440537648042},
}

def normalize_data(pose, stats):
    return [
        (pose[0] - stats['X']['mean']) / stats['X']['std'],
        (pose[1] - stats['Y']['mean']) / stats['Y']['std'],
        (pose[2] - stats['Z']['mean']) / stats['Z']['std'],
        (pose[3] - stats['R']['mean']) / stats['R']['std']
    ]

def denormalize_data(pose, stats):
    return [
        pose[0] * stats['X']['std'] + stats['X']['mean'],
        pose[1] * stats['Y']['std'] + stats['Y']['mean'],
        pose[2] * stats['Z']['std'] + stats['Z']['mean'],
        pose[3] * stats['R']['std'] + stats['R']['mean']
    ]

def load_model(model_path, framework):
    model = None
    if framework == "pytorch":
        model = OpenDoBot()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    return model

def predict(model, image, qpos, lang, framework):
    qpos = normalize_data(qpos, pose_stats)
    qpos = torch.tensor(qpos, dtype=torch.float32).reshape(1, -1).to(device)
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        action = model(qpos, image, lan=lang)
        action = action.squeeze(0).cpu().numpy().tolist()
        return denormalize_data(action, pose_stats)

# --- 主程序 ---
def main():
    model = load_model(MODEL_PATH, FRAMEWORK)
    cap = cv2.VideoCapture(0)
    print("\n摄像头已启动。按 'q' 键退出。按 's' 进行推理。")

    api = dType.load()
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):
        dType.SetQueuedCmdClear(api)
        dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued = 1)
        dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued = 1)
        dType.SetHOMECmd(api, temp = 0, isQueued = 1)

        try:
            lang = None
            qpos = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法从摄像头读取图像。")
                    break

                cv2.imshow("Camera Feed", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("退出程序。")
                    break
                elif key == ord('s'):
                    pose = dType.GetPose(api)
                    qpos = [float(x) for x in pose[:4]]

                    if lang is None:
                        lang = input("请输入语言指令（例如 'pick the green cube'）: ").strip()
                    print(f"使用语言指令: {lang}")
                    print(f"当前输入的 qpos: {qpos}")

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_pil = aug(frame_pil)
                    
                    frame_tensor = transform(frame_pil)

                    action = predict(model, frame_tensor, qpos, lang, FRAMEWORK)
                    print(f"预测动作（反标准化后）: {action}")
                    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, action[0], action[1], action[2], action[3])
            
        finally:
            dType.SetQueuedCmdClear(api)
            dType.SetEndEffectorSuctionCup(api, 0, 1, isQueued = 1)
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头已释放，程序结束。")

    else:
        print("无法连接机械臂")

if __name__ == "__main__":
    main()