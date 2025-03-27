import torchvision
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torchvision import transforms
import librosa
import cv2 as cv


class UM_Audio(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        with open(self.data_root, 'r') as f:
            self.audio_paths = f.readlines()
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x[:, :60000] if x.shape[1] > 60000 else torch.nn.functional.pad(x, (0, 60000 - x.shape[1])))
        ])

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, item):
        # 读取视频路径
        audio_path = self.audio_paths[item][:-1]
        file_name = audio_path.split('/')[-1]
        label_number = int(file_name[7]) - 1

        waveform, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return {'video_data': torch.tensor(mfccs_processed, dtype=torch.float32), 'label': torch.tensor(label_number, dtype=torch.long)}


class UM_Teacher_Data(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        with open(self.data_root, 'r') as f:
            self.audio_paths = f.readlines()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64))
        ])

    def extract_frames(self, video_path, num_frames=20):
        video_data, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        total_frames = video_data.size(0)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        video_data = video_data[frame_indices]
        video_data = video_data / 255.0
        return video_data

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, item):
        # 读取音频和视频路径
        audio_path = self.audio_paths[item][:-1]
        video_path = audio_path.replace('Audio', 'Video')
        file_name = video_path.split('/')[-1]
        video_name = ('01' + file_name[2:]).replace('wav', 'mp4')
        video_path = video_path.replace(file_name, video_name)

        # 制作标签
        label_number = int(video_name[7]) - 1

        # 读取视频数据
        # video_data = self.extract_frames(video_path).permute(3, 0, 1, 2)
        video_data, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        total_frames = video_data.size(0)
        frame_indices = np.linspace(0, total_frames - 1, 20).long()
        video_data = video_data[frame_indices]
        video_data = video_data.permute(3, 0, 1, 2).float()
        video_data = self.transform(video_data) / 255.0

        return {'video_data': video_data, 'label': torch.tensor(label_number, dtype=torch.long)}


class MM_Student(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        with open(self.data_root, 'r') as f:
            self.audio_paths = f.readlines()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64))
        ])

    def extract_frames(self, video_path, num_frames=20):
        video_data, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        frames = []
        total_frames = video_data.size(0)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in frame_indices:
            frame = video_data[idx]
            frame = frame / 255.0
            frames.append(frame)
        return torch.stack(frames)

    def extract_audio(self, audio_path):
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, item):
        # 读取音频和视频路径
        audio_path = self.audio_paths[item][:-1]
        video_path = audio_path.replace('Audio', 'Video')
        file_name = video_path.split('/')[-1]
        video_name = ('01' + file_name[2:]).replace('wav', 'mp4')
        video_path = video_path.replace(file_name, video_name)

        # 制作标签
        label_number = int(video_name[7]) - 1

        # 读取视频数据
        audio_data = self.extract_audio(audio_path)
        audio_data = torch.tensor(audio_data, dtype=torch.float32)

        video_data, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        total_frames = video_data.size(0)
        frame_indices = np.linspace(0, total_frames - 1, 20)
        video_data = video_data[frame_indices]
        video_data = video_data.permute(3, 0, 1, 2).float()
        video_data = self.transform(video_data) / 255.0

        return {'video_data': video_data, 'audio_data': audio_data, 'label': torch.tensor(label_number, dtype=torch.long)}


if __name__ == "__main__":
    teacher_data = UM_Teacher_Data('./data/labeled_sets.txt')
    teacher_dataloader = DataLoader(teacher_data, batch_size=3, shuffle=True)
    for i, _ in enumerate(teacher_dataloader):
        pass
