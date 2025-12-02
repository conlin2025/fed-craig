import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Simple baseline CNN for CIFAR-100
# ----------------------------------------

class SimpleCifarNet(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))   # 16x16
        x = self.pool(F.relu(self.conv3(x)))   # 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ----------------------------------------
# ResNet-18 for CIFAR-100
# ----------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # If input and output dimensions differ (stride or channels), fix via 1x1 conv
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)       # skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()

        self.in_planes = 64

        # CIFAR-specific: first conv is smaller (3x3, stride=1)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet stages
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(
            self.in_planes,
            planes,
            stride
        ))
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_planes,
                planes
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # 32×32
        out = self.layer1(out)                  # 32×32
        out = self.layer2(out)                  # 16×16
        out = self.layer3(out)                  # 8×8
        out = self.layer4(out)                  # 4×4
        out = F.avg_pool2d(out, 4)              # 1×1
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_CIFAR(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# ----------------------------------------
# Factory
# ----------------------------------------

def get_model(num_classes: int = 100,
              device: str = "cpu",
              arch: str = "simple"):
    """
    Return a model on the given device.
    arch:
        - "simple"   : SimpleCifarNet
        - "resnet18" : CIFAR-style ResNet-18
    """
    arch = arch.lower()
    if arch == "resnet18":
        model = ResNet18_CIFAR(num_classes=num_classes)
    elif arch == "simple":
        model = SimpleCifarNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model.to(device)
