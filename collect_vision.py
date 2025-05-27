import cv2
import os

IMG_SAVE_PATH = "imgdata"

if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)
    print(f"目录 '{IMG_SAVE_PATH}' 已创建。")

cap = cv2.VideoCapture(1) # 尝试使用摄像头索引1，如果不行，请尝试0

# 检查是否成功打开了相机
if not cap.isOpened():
    print("无法打开相机。请检查摄像头索引是否正确（尝试0或1）。")
    exit()

print("摄像头已成功打开。")
print("按 's' 键保存当前帧图像。")
print("按 'q' 键退出程序。")

while True:
    # 读取一帧图像
    ret, frame = cap.read()    
    if not ret:
        print("无法读取帧，可能是摄像头已断开。")
        break

    # 显示图像
    cv2.imshow('Camera Feed', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("正在退出...")
        break
    elif key == ord('s'):
        print("\n准备保存图像...")
        while True:
            user_input = input("请输入三个数字作为文件名 (例如: 01 15 30 或 1_2_3): ").strip()
            parts = user_input.replace('_', ' ').split() 

            if len(parts) == 3 and all(part.isdigit() for part in parts):
                try:
                    num1 = int(parts[0])                    
                    num2 = int(parts[1])
                    num3 = int(parts[2])                    
                    filename = f"{num1}_{num2}_{num3}.jpg"
                    filepath = os.path.join(IMG_SAVE_PATH, filename)                    

                    cv2.imwrite(filepath, frame)
                    print(f"图像已成功保存为: {filepath}")
                    break # 成功获取输入并保存后，退出输入循环
                except ValueError:
                    print("输入包含非数字字符，请重新输入。")
                except Exception as e:
                    print(f"保存图像时发生错误: {e}")
                    break # 发生其他错误也退出输入循环
            else:
                print("输入格式不正确。请输入三个由空格或下划线分隔的数字。请重试。")
        print("\n继续捕获图像...")


cap.release()
cv2.destroyAllWindows()
print("摄像头已释放，所有窗口已关闭。")