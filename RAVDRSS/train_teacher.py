from dataload import UM_Teacher_Data, UM_Audio
from models import UMTeacher, UMT_Audio
import torch
from torch.utils.data import DataLoader


def train(seed, train_dataloader, epochs, alpha, beta):
    model = UMTeacher(3, 128, 128, 8, 0., 39).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        acc = 0
        for i, data in enumerate(train_dataloader):
            video_data, labels = data['video_data'].cuda(), data['label'].cuda()
            optimizer.zero_grad()
            video_data = video_data + alpha * abs(video_data).mean() * torch.randn(video_data.shape).cuda()
            pred = model(video_data, beta)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred, 1)
            acc += (predicted == labels).squeeze().sum()
        acc = acc / len(train_dataloader.dataset)
        print('[{}]/[{}] ACC: {}.'.format(epoch+1, epochs, acc))

    torch.save(model.state_dict(), './teacher.pth')
    return 0


train_dataset = UM_Teacher_Data('./data/labeled_sets.txt')
train_dataload = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10)
train(39, train_dataload, 20, 0, 0)

# Audio Acc:100%   Video Acc:14.93%
