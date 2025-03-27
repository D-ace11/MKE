from dataload import UM_Teacher_Data
from model import DeepLabV3Plus
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


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
    model = DeepLabV3Plus(num_classes=41).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(epochs):
        mious = []
        for i, data in enumerate(train_dataloader):
            rgb_data, labels = data['rgb_data'].cuda(), data['label'].cuda()
            optimizer.zero_grad()
            pred = model(rgb_data)
            pred = F.interpolate(pred, labels.shape[1:])
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred, dim=1)
            miou = mean_iou(predicted.cpu().detach().numpy(), labels.cpu().detach().numpy(), num_classes=41)
            mious.append(miou)
            print('Epoch [{}]: {}'.format(epoch+1, miou))
        print('Epoch [{}]/[{}] mIoU: {}.'.format(epoch+1, epochs, np.array(mious).mean()))

    torch.save(model.state_dict(), './teacher.pth')
    return 0


train_dataset = UM_Teacher_Data('./data/train.txt')
train_dataload = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=10)
train(train_dataload, 500)

# mIoU: 65.47
