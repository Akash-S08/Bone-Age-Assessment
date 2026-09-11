import torch
import torch.nn as nn
import torchvision.models as models


# ---------- CBAM ----------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, mx], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        att = self.sa(x)
        x = att * x
        return x, att


# ---------- MAIN MODEL ----------
class BoneAgeNet(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True):
        super().__init__()

        base = models.resnet34(pretrained=pretrained)
        feat_dim = 512

        # SAME AS TRAINING
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        # ADD CBAM BACK
        self.attention = CBAM(feat_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)

        feat, att = self.attention(feat)

        pooled = self.pool(feat)
        pooled = pooled.view(pooled.size(0), -1)

        age = self.classifier(pooled)

        return {
            "age_pred": age,
            "attention_map": att,
            "features": feat
        }


def create_model(config):
    return BoneAgeNet(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"]
    )