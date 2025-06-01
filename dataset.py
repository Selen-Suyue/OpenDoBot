import xml.etree.ElementTree as ET
import os
import glob # For finding files matching a pattern
from PIL import Image # For loading images
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms # Optional, for image transformations
import torchvision.transforms as T

# --- Your provided playback parser ---
def parse_playback_file(filepath):
    """
    解析 playback 文件并提取 X, Y, Z, R 数据。

    Args:
        filepath (str): .playback 文件的路径。

    Returns:
        list: 包含每个时刻数据的字典列表，如果出错则返回 None。
              每个字典格式为: {'时刻索引': int, 'X': float, 'Y': float, 'Z': float, 'R': float}
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件 '{filepath}' 不存在。")
        return None

    all_moments_data = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        moment_elements = [
            el for el in root
            if el.tag.startswith('row') and el.tag[3:].isdigit()
        ]

        if not moment_elements:
            print(f"警告: 在文件 '{filepath}' 中没有找到 'rowX' 格式的时刻元素。")
            return []

        for i, moment_element in enumerate(moment_elements):
            current_moment_values = {'时刻索引': i}
            keys_items = {'X': 'item_2', 'Y': 'item_3', 'Z': 'item_4', 'R': 'item_5'}
            valid_moment = True
            for key, item_tag in keys_items.items():
                element = moment_element.find(item_tag)
                if element is not None and element.text is not None:
                    try:
                        current_moment_values[key] = float(element.text)
                    except ValueError:
                        current_moment_values[key] = None
                        print(f"警告: 时刻 {i}, 文件 '{filepath}', {item_tag} ('{element.text}') 无法转换为浮点数。")
                        valid_moment = False # Mark moment as potentially problematic if any value is None
                else:
                    current_moment_values[key] = None
                    valid_moment = False # Mark moment as potentially problematic

            # Only add if all essential coordinates are present (X, Y, Z, R)
            # You might want to adjust this check based on how critical each coordinate is
            if all(current_moment_values.get(k) is not None for k in ['X', 'Y', 'Z', 'R']):
                all_moments_data.append(current_moment_values)
            else:
                print(f"警告: 时刻 {i} 在文件 '{filepath}' 中缺少一个或多个坐标数据，已跳过。数据: {current_moment_values}")

        return all_moments_data

    except ET.ParseError as e:
        print(f"错误: 解析 XML 文件 '{filepath}' 失败。错误信息: {e}")
        return None
    except Exception as e:
        print(f"处理文件 '{filepath}' 时发生未知错误: {e}")
        return None

# --- PyTorch Dataset Class ---
class RobotImitationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.demos_dir = os.path.join(root_dir, "demos")
        self.img_dir = os.path.join(root_dir, "imgdata")
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        self.aug = T.Compose([
                T.RandomPerspective(distortion_scale=0.2),
                T.RandomResizedCrop((256,256), scale=(0.6, 1.0), ratio=(1.0, 1.0)),
                T.RandomApply([
                T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1
                )
                ], p=0.4)
                ])
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        playback_files = glob.glob(os.path.join(self.demos_dir, "*.playback"))

        for playback_file_path in playback_files:
            filename = os.path.basename(playback_file_path)
            # 提取 task_id 和 demo_id, e.g., from "1_1.playback" -> task_demo_id = "1_1"
            # "initial.playback" might be special, handle if needed or it will be skipped
            # if no images like initial_X_Y.jpg exist.
            if filename == "initial.playback": # Example: skipping initial.playback
                print(f"跳过特殊文件: {filename}")
                continue
            
            try:
                task_demo_id = filename.split('.')[0] # "1_1"
                task_id_str = task_demo_id.split('_')[0] # "1"
            except IndexError:
                print(f"警告: 无法从文件名 '{filename}' 解析 task_id 和 demo_id。已跳过。")
                continue

            robot_positions_data = parse_playback_file(playback_file_path)

            if not robot_positions_data: # Handles None or empty list
                print(f"警告: 未能从 '{playback_file_path}' 加载机器人位置数据，或数据为空。")
                continue

            num_timesteps = len(robot_positions_data)
            if num_timesteps == 0:
                continue

            for t in range(num_timesteps):
                current_pos_data = robot_positions_data[t]
                
                # 确保所有坐标都已成功解析
                if any(current_pos_data.get(coord) is None for coord in ['X', 'Y', 'Z', 'R']):
                    print(f"警告: 在 {task_demo_id} 的时间步 {t} 缺少机器人位置数据。已跳过此时间步。")
                    continue

                current_robot_pos = [
                    current_pos_data['X'],
                    current_pos_data['Y'],
                    current_pos_data['Z'],
                    current_pos_data['R']
                ]

                # 构造图像路径
                # 图像名称格式: TASKID_DEMOID_TIMESTEP.jpg, e.g., 1_1_0.jpg
                # playback 的 '时刻索引' (t)直接对应图像的 TIMESTEP
                img_filename = f"{task_demo_id}_{t}.jpg"
                img_path = os.path.join(self.img_dir, img_filename)

                if not os.path.exists(img_path):
                    print(f"警告: 图像文件 '{img_path}' 不存在，对应 {task_demo_id} 的时间步 {t}。已跳过此样本。")
                    continue

                # 确定动作 (下一时间步的位置)
                if t < num_timesteps - 1:
                    next_pos_data = robot_positions_data[t+1]
                    if any(next_pos_data.get(coord) is None for coord in ['X', 'Y', 'Z', 'R']):
                        print(f"警告: 在 {task_demo_id} 的时间步 {t+1} (作为动作) 缺少机器人位置数据。已跳过此样本。")
                        continue
                    action = [
                        next_pos_data['X'],
                        next_pos_data['Y'],
                        next_pos_data['Z'],
                        next_pos_data['R']
                    ]
                else: # 最后一个时间步，动作是保持当前位置
                    action = current_robot_pos

                self.samples.append({
                    "task_id": task_id_str, # "1"
                    "task_demo_id": task_demo_id, # "1_1" for debugging or more specific task identification
                    "image_path": img_path,
                    "current_pos": current_robot_pos,
                    "action": action,
                    "timestep": t # For debugging
                })
        
        print(f"成功加载 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_info = self.samples[idx]

        task_id = sample_info["task_id"]
        image_path = sample_info["image_path"]
        current_pos = torch.tensor(sample_info["current_pos"], dtype=torch.float32)
        action = torch.tensor(sample_info["action"], dtype=torch.float32)

        try:
            image = Image.open(image_path).convert('RGB') # Ensure 3 channels
        except FileNotFoundError:
            print(f"错误: 在 __getitem__ 中找不到图像文件: {image_path}")
            # Handle error appropriately, e.g., return a placeholder or raise exception
            # For now, let's try to return the next valid sample or raise error
            # This should ideally be caught during _load_samples, but as a safeguard:
            if idx + 1 < len(self.samples):
                return self.__getitem__(idx + 1)
            else:
                raise FileNotFoundError(f"Image not found: {image_path} and no more samples.")
        except Exception as e:
            print(f"错误: 加载图像时出错 {image_path}: {e}")
            if idx + 1 < len(self.samples):
                return self.__getitem__(idx + 1)
            else:
                raise RuntimeError(f"Error loading image: {image_path} and no more samples.")


        
        image = self.aug(image)
        image = self.transform(image)
        
        return {
            "task_id": task_id, # e.g., "1"
            "image": image,
            "current_pos": current_pos,
            "action": action
        }

# --- 自定义 collate 函数 ---
def custom_collate(batch):
    """
    自定义collate函数，将所有数据转换为指定形状的张量
    
    返回:
        dict: 包含以下键的字典:
            - task_id: 形状为 (batch_size, 1) 的整数张量
            - image: 形状为 (batch_size, 3, 256, 256) 的浮点张量
            - current_pos: 形状为 (batch_size, 4) 的浮点张量
            - action: 形状为 (batch_size, 4) 的浮点张量
    """
    # 处理任务ID: 字符串 -> 整数 -> 张量
    task_ids = [int(item['task_id']) for item in batch]
    task_ids_tensor = torch.tensor(task_ids, dtype=torch.long).unsqueeze(1)  # (batch_size, 1)
    
    # 处理图像: 已经是(3, 256, 256)的张量，直接堆叠
    images = torch.stack([item['image'] for item in batch])  # (batch_size, 3, 256, 256)
    
    # 处理位置和动作: 已经是(4)的张量，直接堆叠
    current_positions = torch.stack([item['current_pos'] for item in batch])  # (batch_size, 4)
    actions = torch.stack([item['action'] for item in batch])  # (batch_size, 4)
    
    return {
        'task_id': task_ids_tensor,
        'image': images,
        'current_pos': current_positions,
        'action': actions
    }

# --- 主程序 ---
if __name__ == "__main__":
    # 创建数据集实例
    robot_dataset = RobotImitationDataset(root_dir="data")
    print(f"数据集中的样本数量: {len(robot_dataset)}")
    
    if len(robot_dataset) > 0:
        batch_size = len(robot_dataset)
        dataloader = DataLoader(
            robot_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=0
        )
        
        full_batch = next(iter(dataloader))
        
        # 打印批处理结果
        print("\n完整批处理结果:")
        print(f"  任务ID张量形状: {full_batch['task_id'].shape} (数据类型: {full_batch['task_id'].dtype})")
        print(f"  图像张量形状: {full_batch['image'].shape} (数据类型: {full_batch['image'].dtype})")
        print(f"  当前位置张量形状: {full_batch['current_pos'].shape} (数据类型: {full_batch['current_pos'].dtype})")
        print(f"  动作张量形状: {full_batch['action'].shape} (数据类型: {full_batch['action'].dtype})")
        
        sample1_size, sample2_size = 0, 0
        for sample_idx in range(batch_size):
            if (full_batch['task_id'][sample_idx].item() == 1):
                sample1_size += 1
            elif (full_batch['task_id'][sample_idx].item() == 2):
                sample2_size += 1
            else:
                print(f"警告: 未知任务 ID {full_batch['task_id'][sample_idx].item()} 在样本索引 {sample_idx} 中。")

        print(f"  任务 ID 1 的样本数量: {sample1_size}")
        print(f"  任务 ID 2 的样本数量: {sample2_size}")

        for sample_idx in range(10):
            print(f"批次中第{sample_idx}个样本的详细信息:")
            print(f"  任务 ID: {full_batch['task_id'][sample_idx].item()} (原始值)")
            print(f"  图像 Tensor 形状: {full_batch['image'][sample_idx].shape}")
            print(f"  当前机器人位置: {full_batch['current_pos'][sample_idx]}")
            print(f"  机器人动作: {full_batch['action'][sample_idx]}")
            

    else:
        print("数据集中没有样本。请检查数据路径和文件格式。")
    
    print("------------------------------")