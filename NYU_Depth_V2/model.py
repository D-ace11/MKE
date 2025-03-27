import torch.nn as nn
import torch
from torchvision.models import resnet101
from torchvision.models._utils import IntermediateLayerGetter


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        for r in rates:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        self.blocks.append(nn.Sequential(  # global average pooling branch
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        out = [block(x) for block in self.blocks[:-1]]
        global_feat = self.blocks[-1](x)
        global_feat = nn.functional.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        out.append(global_feat)
        out = torch.cat(out, dim=1)
        return self.project(out)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        torch.manual_seed(39)
        backbone = resnet101(weights='DEFAULT', replace_stride_with_dilation=[False, True, True])
        self.backbone = IntermediateLayerGetter(backbone, return_layers={'layer4': 'out', 'layer1': 'low'})

        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.backbone(x)

        low_level_feat = self.low_level_proj(features['low'])
        high_level_feat = self.aspp(features['out'])

        high_level_feat = nn.functional.interpolate(high_level_feat, size=low_level_feat.shape[2:], mode='bilinear',
                                                    align_corners=False)
        x = torch.cat([high_level_feat, low_level_feat], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


if __name__ == '__main__':
    model = DeepLabV3Plus(num_classes=41)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
