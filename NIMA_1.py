# 作   者:BZ
# 开发时间:2025/3/3
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models.inception import InceptionOutputs
from utils_1 import uti as normalized


class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        base_model = vgg16(pretrained=True)
        base_model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        base_model.aux_logits = False
        base_model.AuxLogits = None

        base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 10)
        )

        self.base_model = base_model

    def forward(self, x):
        outputs = self.base_model(x)
        if isinstance(outputs, InceptionOutputs):
            outputs = outputs.logits
        return torch.softmax(outputs, dim=1)


class EarthMoverDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        cdf_pred = torch.cumsum(pred, dim=1)
        cdf_target = torch.cumsum(target, dim=1)
        sub = (cdf_pred - cdf_target) ** 2
        mean = torch.mean(sub, dim=1)
        sq = torch.sqrt(mean)
        return torch.mean(sq)
