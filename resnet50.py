import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        mid_channels = out_channels//4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=(2 if downsample else 1))
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        if downsample:
            self.downsmaple = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        elif in_channels != out_channels:
            self.downsmaple = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsmaple = None

    def forward(self, x):
        residual = x
        if self.downsmaple:
            residual = self.downsmaple(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = x + residual
        x = self.act(x)

        return x
    
class ResNet50(nn.Module):
    def __init__(self, class_num):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            BottleNeck(64, 256),
            BottleNeck(256, 256),
            BottleNeck(256, 256)
        )
        self.conv3 = nn.Sequential(
            BottleNeck(256, 512, downsample=True),
            BottleNeck(512, 512),
            BottleNeck(512, 512),
            BottleNeck(512, 512)
        )
        self.conv4 = nn.Sequential(
            BottleNeck(512, 1024, downsample=True),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024),
            BottleNeck(1024, 1024)
        )
        self.conv5 = nn.Sequential(
            BottleNeck(1024, 2048, downsample=True),
            BottleNeck(2048, 2048),
            BottleNeck(2048, 2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, class_num)
    
    def forward(self, x):
        # conv1 & maxpool
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.maxpool(x)
        # conv2
        x = self.conv2(x)

        # conv3
        x = self.conv3(x)

        # conv4
        x = self.conv4(x)

        # conv5
        x = self.conv5(x)

        # avg pool
        x = self.avgpool(x)

        b, c, _, _ =  x.shape
        x = torch.reshape(x, [b, c])
        x = self.fc(x)
        
        return x
    
if __name__ == "__main__":
    import torch

    model = ResNet50().to('cuda')
    sample_img = torch.randn(32, 3, 224, 224, device='cuda')
    out = model(sample_img)
    print(out.shape)