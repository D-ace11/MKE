import torch
import torch.nn as nn


class UMT_Audio(nn.Module):
    def __init__(self, audio_channel=3, hidden_dim=128, mlp_dim=128, output_dim=8, dropout_rate=0., seed=0):
        super().__init__()
        torch.manual_seed(seed)
        # video CNN
        self.video_cnn = nn.Sequential(
            nn.Conv1d(audio_channel, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # MLP
        self.fc1 = nn.Linear(128*5, mlp_dim)
        self.fc2 = nn.Linear(hidden_dim, mlp_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, audio, beta):
        audio = audio.unsqueeze(1)
        audio_feature = self.video_cnn(audio).squeeze(-1).squeeze(-1).squeeze(-1)
        audio_feature = self.dropout(self.relu((self.fc1(audio_feature.view(4, -1)))))
        audio_feature = audio_feature + beta * abs(audio_feature).mean() * torch.randn(audio_feature.shape).cuda()
        audio_feature = self.dropout(self.relu((self.fc2(audio_feature))))
        output = self.fc3(audio_feature)
        return output


class UMTeacher(nn.Module):
    def __init__(self, video_channel=3, hidden_dim=128, mlp_dim=128, output_dim=8, dropout_rate=0., seed=0):
        super().__init__()
        torch.manual_seed(seed)
        # video CNN
        self.video_cnn = nn.Sequential(
            nn.Conv3d(video_channel, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        # MLP
        self.fc1 = nn.Linear(hidden_dim * 8 * 8 * 2, mlp_dim)
        self.fc2 = nn.Linear(hidden_dim, mlp_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, video, beta):
        video_feature = self.video_cnn(video).view(video.size(0), -1)
        video_feature = self.dropout(self.relu((self.fc1(video_feature))))
        video_feature = video_feature + beta * abs(video_feature).mean() * torch.randn(video_feature.shape).cuda()
        video_feature = self.dropout(self.relu((self.fc2(video_feature))))
        output = self.fc3(video_feature)
        return output


class MMStudent(nn.Module):
    def __init__(self, video_channel=3, audio_channel=1, hidden_dim=128, mlp_dim=128, output_dim=8, dropout_rate=0., seed=0):
        super().__init__()
        torch.manual_seed(seed)
        # video CNN
        self.video_cnn = nn.Sequential(
            nn.Conv3d(video_channel, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        # audio CNN
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(audio_channel, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # MLP
        # self.fc1 = nn.Linear(hidden_dim * 5 + 8 * 8 * 2 * hidden_dim, mlp_dim)
        self.fc1 = nn.Linear(hidden_dim * 5 + hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(hidden_dim, mlp_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, video, audio, beta):
        audio = audio.unsqueeze(1)
        video_feature = self.video_cnn(video).view(video.size(0), -1)
        audio_feature = self.audio_cnn(audio).view(audio.size(0), -1)

        combine_feat = torch.cat([video_feature, audio_feature], dim=-1)
        combine_feat = self.dropout(self.relu((self.fc1(combine_feat))))
        combine_feat = combine_feat + beta * abs(combine_feat).mean() * torch.randn(combine_feat.shape).cuda()
        combine_feat = self.dropout(self.relu((self.fc2(combine_feat))))
        output = self.fc3(combine_feat)
        return output


if __name__ == "__main__":
    # (batch_size, channels, frames, height, width)
    video = torch.randn(8, 3, 16, 112, 112)
    # (batch_size, channels, samples)
    audio = torch.randn(8, 1, 16000)

    mm = MMStudent()
    output = mm(video, audio)
    print(output.shape)