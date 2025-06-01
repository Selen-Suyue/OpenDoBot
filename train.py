from torch.utils.data import Dataset, DataLoader
from dataset import CustomDataset, custom_collate

if __name__ == "__main__":

    robot_dataset = CustomDataset(root_dir=root_dir)
    batch_size = len(robot_dataset)
    dataloader = DataLoader(
        robot_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )

    full_batch = next(iter(dataloader))
