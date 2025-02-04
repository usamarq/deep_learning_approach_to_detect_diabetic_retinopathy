import copy
import torch
import torch.nn as nn
from torchvision import models
from AttentionMechanisms import SelfAttention, ChannelAttention, SpatialAttention

# ResNet-18
# ------------------------------------------------------------------------------------------ 
class Resnet18_Single(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.gradients = None
        self.activations = None  # To store feature maps

        # Register hooks
        self.backbone.layer4[1].register_full_backward_hook(self.save_gradient)
        self.backbone.layer4[1].register_forward_hook(self.save_activation)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output  # Save activations from the target layer

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class Resnet18_Dual(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()

        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

class Resnet18_Single_SpatialAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.attention = SpatialAttention()
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape to [batch_size, 512, 1, 1] for attention
        attention_map = self.attention(x)  # Apply spatial attention
        x = x * attention_map  # Element-wise multiplication
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x
    
class Resnet18_Single_ChannelAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.attention = ChannelAttention(in_channels=512)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        attention_map = self.attention(x)
        x = x * attention_map
        x = self.fc(x)
        return x
    
class Resnet18_Single_SelfAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.attention = SelfAttention(in_channels=512)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.fc(x)
        return x

class Resnet18_Single_Pretrained(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/resnet18.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
        print('unexpected keys:', info[1])

        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
class Resnet18_Single_Pretrained_SpacialAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        
        self.backbone = models.resnet18(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/resnet18.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
        print('unexpected keys:', info[1])
        self.attention = SpatialAttention()
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape to [batch_size, 512, 1, 1] for attention
        attention_map = self.attention(x)  # Apply spatial attention
        x = x * attention_map  # Element-wise multiplication
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x

class Resnet18_Single_Pretrained_ChannelAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/resnet18.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
        print('unexpected keys:', info[1])
        self.attention = ChannelAttention(in_channels=512)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape to [batch_size, 512, 1, 1] for attention
        attention_map = self.attention(x)  # Apply spatial attention
        x = x * attention_map  # Element-wise multiplication
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x
    
class Resnet18_Single_Pretrained_SelfAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/resnet18.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
        print('unexpected keys:', info[1])
        self.attention = SelfAttention(in_channels=512)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Reshape to [batch_size, 512, 1, 1] for attention
        attention_map = self.attention(x)  # Apply spatial attention
        x = x * attention_map  # Element-wise multiplication
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x

# ResNet-34
# ------------------------------------------------------------------------------------------ 

class Resnet34_Single(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.gradients = None
        self.activations = None  # To store feature maps

        # Register hooks
        self.backbone.layer4[1].register_full_backward_hook(self.save_gradient)
        self.backbone.layer4[1].register_forward_hook(self.save_activation)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output  # Save activations from the target layer

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class Resnet34_Single_Pretrained(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        state_dict = torch.load('pretrained_DR_resize/pretrained/resnet34.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
        print('unexpected keys:', info[1])
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    

# VGG16
# ------------------------------------------------------------------------------------------ 

class VGG16_Single(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.vgg16(pretrained=True)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone.classifier = nn.Identity() # Remove the original classification layer
        
        # self.gradients = None
        # self.activations = None  # To store feature maps

        # # Register hooks
        # self.backbone.layer4[1].register_full_backward_hook(self.save_gradient)
        # self.backbone.layer4[1].register_forward_hook(self.save_activation)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
    

    # def save_gradient(self, module, grad_input, grad_output):
    #     self.gradients = grad_output[0]

    # def save_activation(self, module, input, output):
    #     self.activations = output  # Save activations from the target layer

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
class VGG16_Single_Pretrained(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = models.vgg16(pretrained=True)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        state_dict = torch.load('pretrained_DR_resize/pretrained/vgg16.pth', map_location='cpu')
        info = self.backbone.load_state_dict(state_dict, strict=False)
        print('missing keys:', info[0])  # The missing fc or classifier layer is normal here
        print('unexpected keys:', info[1])
        self.backbone.classifier = nn.Identity()  # Remove the original classification layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
