# 作   者:BZ
# 开发时间:2025/3/3
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import SqueezeExcitation


# --------------------------------------
# 模型定义
# --------------------------------------
class NIMA(nn.Module):
    def __init__(self, pretrained=True, width_mult=1.0, dropout=0.2):
        super().__init__()

        self.width_mult = width_mult
        self.dropout = dropout

        base_model = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        )

        self.features = self._customize_features(base_model.features)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.out_channels = self.features(dummy).shape[1]

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(int(self.out_channels * width_mult), 256),
            nn.Hardswish(inplace=True),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def _customize_features(self, original_features):
        layers = []

        first_conv = nn.Sequential(
            nn.Conv2d(3, int(16 * self.width_mult), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(16 * self.width_mult)),
            nn.Hardswish(inplace=True)
        )
        layers.append(first_conv)

        # 修改中间层配置
        custom_blocks = [
            # 格式：(kernel_size, expansion_size, output_channels, SE_ratio, stride)
            (3, 1, 16, 0.25, 1),  # 修改后的初始层
            (3, 4, 24, 0.25, 2),  # 减小expansion ratio
            (3, 4, 24, 0.25, 1),
            (5, 4, 48, 0.25, 2),  # 使用5x5卷积增强特征捕获
            (5, 6, 40, 0.25, 1),
            (5, 6, 48, 0.25, 1),
            (5, 6, 40, 0.25, 1),
            (3, 6, 96, 0.25, 2),  # 深层保持原结构
            (3, 6, 96, 0.25, 1),
            (3, 6, 96, 0.25, 1)
        ]

        in_channels = int(16 * self.width_mult)
        for idx, (k, exp, out_c, se_ratio, s) in enumerate(custom_blocks):
            out_channels = int(out_c * self.width_mult)
            expansion = int(exp * in_channels)

            layers.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    stride=s,
                    expansion=expansion,
                    se_ratio=se_ratio
                )
            )
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        return self.regressor(features)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_res = stride == 1 and in_channels == out_channels

        expanded_channels = expansion
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.act1 = nn.Hardswish(inplace=True)

        self.conv2 = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=(kernel_size - 1) // 2,
            groups=expanded_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.act2 = nn.Hardswish(inplace=True)

        self.se = SqueezeExcitation(
            expanded_channels,
            squeeze_channels=max(1, int(expanded_channels * se_ratio))
        ) if se_ratio > 0 else nn.Identity()

        self.conv3 = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.use_res:
            out += residual
        return out

class EarthMoverDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        cdf_pred = torch.cumsum(pred, dim=1)
        cdf_target = torch.cumsum(target, dim=1)
        return torch.mean(torch.sqrt(torch.mean((cdf_pred + cdf_target) ** 2, dim=1)))
