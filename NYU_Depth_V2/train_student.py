from dataload import MM_Student
from stu_model import DeepLab
from model import DeepLabV3Plus
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def mean_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            iou = float('nan') # 避免除零错误
        else:
            iou = intersection / union

        ious.append(iou)

    miou = np.nanmean(ious) # 忽略nan类别
    return miou


def train(train_dataloader, epochs):
    model = DeepLab(41, criterion=nn.CrossEntropyLoss(),
                pretrained_model='./resnet101_v1c.pth',
                norm_layer=nn.BatchNorm2d).cuda()
    teacher_model = DeepLabV3Plus(num_classes=41).cuda()
    state_dict = torch.load('./teacher.pth')
    teacher_model.load_state_dict(state_dict)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    teacher_model.eval()
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
    for epoch in range(epochs):
        mious = []
        for i, data in enumerate(train_dataloader):
            rgb_data, hha_data, labels = data['rgb_data'].cuda(), data['hha_data'].cuda(), data['label'].cuda()
            optimizer.zero_grad()

            with torch.no_grad():
                pl = teacher_model(rgb_data)

            pred = model(rgb_data, hha_data)
            _, pl = torch.max(pl, dim=1)
            # loss = criterion(pred, pl) * 0.1 + criterion(F.interpolate(pred, labels.shape[1:]), labels)
            loss = criterion(pred, pl)
            loss.backward()
            optimizer.step()

            pred = F.interpolate(pred, labels.shape[1:])
            _, predicted = torch.max(pred, dim=1)
            miou = mean_iou(predicted.cpu().detach().numpy(), labels.cpu().detach().numpy(), num_classes=41)
            mious.append(miou)
            print('Epoch [{}]: {}'.format(epoch+1, miou))
        print('Epoch [{}]/[{}] mIoU: {}.'.format(epoch+1, epochs, np.array(mious).mean()))

    torch.save(model.state_dict(), './student.pth')
    return 0


train_dataset = MM_Student('./data/test.txt')
train_dataload = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=10)
train(train_dataload, 500)

# mIoU: 77.24
