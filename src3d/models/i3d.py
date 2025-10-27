import torch #type: ignore
import torch.nn as nn #type: ignore

class InceptionModule3D(nn.Module):
    """3D Inception Module for I3D"""
    def __init__(self, in_channels, out_channels):
        super(InceptionModule3D, self).__init__()
        
        
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[0], kernel_size=1),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[1], kernel_size=1),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels[1], out_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels[3], kernel_size=1),
            nn.BatchNorm3d(out_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels[3], out_channels[4], kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels[4], out_channels[5], kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels[5]),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels[6], kernel_size=1),
            nn.BatchNorm3d(out_channels[6]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class I3D(nn.Module):
    """
    I3D (Inflated 3D ConvNet)
    State-of-the-art architecture for video classification
    """
    def __init__(self, num_classes=2288, dropout=0.5):
        super(I3D, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        self.inception3a = InceptionModule3D(192, [64, 96, 128, 16, 32, 32, 32])
        self.inception3b = InceptionModule3D(256, [128, 128, 192, 32, 96, 96, 64])
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.inception4a = InceptionModule3D(480, [192, 96, 208, 16, 48, 48, 64])
        self.inception4b = InceptionModule3D(512, [160, 112, 224, 24, 64, 64, 64])
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.maxpool4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
