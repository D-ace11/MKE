import torchvision
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torchvision import transforms
import librosa
from PIL import Image


class UM_Teacher_Data(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        with open(self.data_root, 'r') as f:
            self.rgb_paths = f.readlines()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, item):
        # 读取路径
        rgb_path = self.rgb_paths[item][:-1]
        label_path = rgb_path.replace('RGB', 'labels').replace('jpg', 'png')

        # 读取数据
        rgb_img = Image.open(rgb_path)
        label_img = Image.open(label_path)
        rgb_img, label_img = np.array(rgb_img) / 255., np.array(label_img)
        rgb_img = self.transform(rgb_img)

        return {'rgb_data': rgb_img.float(), 'label': torch.tensor(label_img, dtype=torch.int64)}


class MM_Student(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        with open(self.data_root, 'r') as f:
            self.rgb_paths = f.readlines()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, item):
        # 读取路径
        rgb_path = self.rgb_paths[item][:-1]
        hha_path = rgb_path.replace('RGB', 'HHA').replace('jpg', 'png')
        label_path = rgb_path.replace('RGB', 'labels').replace('jpg', 'png')

        # 读取数据
        rgb_img = Image.open(rgb_path)
        hha_img = Image.open(hha_path)
        label_img = Image.open(label_path)
        rgb_img, hha_img, label_img = np.array(rgb_img) / 255., np.array(hha_img) / 255.,  np.array(label_img)
        rgb_img = self.transform(rgb_img)
        hha_img = self.transform(hha_img)

        return {'rgb_data': rgb_img.float(), 'hha_data': hha_img.float(), 'label': torch.tensor(label_img, dtype=torch.int64)}


if __name__ == "__main__":
    teacher_data = UM_Teacher_Data('./data/train.txt')
    teacher_dataloader = DataLoader(teacher_data, batch_size=4, shuffle=True)
    for i, _ in enumerate(teacher_dataloader):
        pass
