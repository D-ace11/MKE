from dataload import MM_Student
from models import MMStudent, UMTeacher
import torch
from torch.utils.data import DataLoader


def train(seed, train_dataloader, epochs, alpha, beta):
    stu_model = MMStudent(3, 1, 128, 128, 8, 0., 39).cuda()
    tea_model = UMTeacher(3, 128, 128, 8, 0, 39).cuda()
    state_dict = torch.load('./teacher.pth')
    tea_model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(stu_model.parameters(), lr=0.01)
    stu_model.train()
    tea_model.eval()
    for epoch in range(epochs):
        acc = 0
        for i, data in enumerate(train_dataloader):
            video_data, audio_data, labels = data['video_data'].cuda(), data['audio_data'].cuda(), data['label'].cuda()
            pl = tea_model(video_data, beta)
            optimizer.zero_grad()
            video_data = video_data + alpha * abs(video_data).mean() * torch.randn(video_data.shape).cuda()
            audio_data = audio_data + alpha * abs(audio_data).mean() * torch.randn(audio_data.shape).cuda()

            _, pl = torch.max(pl, 1)
            pre = stu_model(video_data, audio_data, beta)
            loss = criterion(pre, pl)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pre, 1)
            acc += (predicted == labels).squeeze().sum()
        acc = acc / len(train_dataloader.dataset)
        print('[{}]/[{}] ACC: {}.'.format(epoch+1, epochs, acc))

    torch.save(stu_model.state_dict(), './student.pth')
    return 0


train_dataset = MM_Student('./data/unlabeled_train.txt')
train_dataload = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
train(39, train_dataload, 20, 1, 0)

# Audio Acc:100%   Video Acc:15.27%
