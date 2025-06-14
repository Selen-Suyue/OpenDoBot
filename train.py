import torch
import os
from policy import OpenDoBot, OpenDoBot_v2
from torch.utils.data import DataLoader
from dataset import RobotImitationDataset, custom_collate, denormalize_data
from transformers import get_cosine_schedule_with_warmup

from tqdm import tqdm

def train():
    num_epochs=1000
    batch_size=80
    save_epochs=50
    ckpt_dir = 'logs/depth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    robot_dataset = RobotImitationDataset(root_dir='data')
    dataloader = DataLoader(
        robot_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=4
    )

    policy = OpenDoBot().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    train_history=[]
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)
        num_steps = len(dataloader)
        optimizer.zero_grad()
        avg_loss = 0.0
        for data in pbar:
            action = data['action'].to(device)
            qpos = data['current_pos'].to(device)
            img = data['image'].to(device)
            task_id = data['task_id'].to(device)

            noise_mask = torch.rand(qpos.size(0), device=qpos.device) < 0.3
            noise = torch.randn_like(qpos) 
            qpos[noise_mask] = noise[noise_mask]

            loss = policy(qpos, img, id=task_id,actions=action)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()
        avg_loss = avg_loss / num_steps
        train_history.append(avg_loss)


        print("Train loss: {:.6f}".format(avg_loss))

        if (epoch + 1) % save_epochs == 0:
                torch.save(
                    policy.state_dict(),
                    os.path.join(ckpt_dir, "depth_policy_epoch_{}.ckpt".format(epoch + 1))
                )
                print(train_history)

if __name__ == "__main__":
    train()

