#import libraries
import torch.nn as nn
from torchvision.models import resnet18


#locked model axis, cifar-10 resnet-18
#3x3 stride-1 stem, drop maxpool, keep 32x32 detail
def build_model(num_classes: int = 10) -> nn.Module:
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
