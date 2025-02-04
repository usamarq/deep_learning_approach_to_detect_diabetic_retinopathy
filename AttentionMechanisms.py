import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        # Average pooling and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel dimension
        x = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution
        x = self.conv1(x)
        return torch.sigmoid(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        # Global Average Pooling
        b, c, _, _ = x.size()
        avg_out = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_out = F.adaptive_max_pool2d(x, 1).view(b, c)

        # Channel Attention
        avg_out = self.fc2(F.relu(self.fc1(avg_out)))
        max_out = self.fc2(F.relu(self.fc1(max_out)))

        # Combine the two paths
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # Compute query, key, and value
        query = self.query_conv(x).view(batch_size, -1, width * height)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        # Compute attention scores
        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key) / (C ** 0.5), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)

        return self.gamma * out + x  # Residual connection