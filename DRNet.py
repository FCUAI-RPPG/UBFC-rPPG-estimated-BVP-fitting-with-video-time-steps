import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.upconv(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SpatialAttentionBlock(nn.Module):
    def __init__(self, channels,H):
        super(SpatialAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.avg_pool  = nn.AdaptiveAvgPool2d((None, 1))
        self.fc1 = nn.Linear(H*channels, H*channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(H*channels, H*channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        residual = x
        residual = self.avg_pool(residual)
        batch_size, channels, height, _ = residual.shape
        residual = residual.view(-1)
        residual = self.fc1(residual)
        residual = residual.view(batch_size, channels, height,-1)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        residual = residual.view(-1)
        residual = self.fc2(residual)
        residual = residual.view(batch_size, channels, height,-1)
        residual = self.sigmoid(residual)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return residual * x

class PhysiologicalEstimator(nn.Module):
    def __init__(self):
        super(PhysiologicalEstimator, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.down1 = DownBlock(3, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)
        self.down5 = DownBlock(256, 512)
        
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.up5 = UpBlock(32, 1)
        
        self.spatial_attention1 = SpatialAttentionBlock(3,32)
        self.spatial_attention2 = SpatialAttentionBlock(32,32)
        self.spatial_attention3 = SpatialAttentionBlock(64,32)
        self.spatial_attention4 = SpatialAttentionBlock(128,32)
        self.spatial_attention5 = SpatialAttentionBlock(256,32)
        
    def forward(self, x):
        x = self.spatial_attention1(x)
        x = self.down1(x)
        x = self.spatial_attention2(x)
        x = self.down2(x)
        x = self.spatial_attention3(x)
        x = self.down3(x)
        x = self.spatial_attention4(x)
        x = self.down4(x)
        x = self.spatial_attention5(x)
        x = self.down5(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x

# Example usage
model = PhysiologicalEstimator()
input_tensor = torch.randn(1, 3, 32, 256)  # Example input tensor
start_time = time.perf_counter()
output = model(input_tensor)
print((time.perf_counter()-start_time)/256)

# model.to(torch.device("cuda"))
# from torchsummary import summary
# summary(model,input_size = (3, 32, 256))
