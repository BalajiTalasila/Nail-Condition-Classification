import torch
import torch.nn as nn
import timm


class SpatialAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // 4,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // 4,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        attention_map = self.attention(x)

        return x * attention_map


class AttentionEfficientNetB0(nn.Module):

    def __init__(
        self,
        num_classes=6
    ):

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        channels = self.backbone.num_features

        self.attention = SpatialAttention(
            channels
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            channels,
            num_classes
        )

    def forward(self, x):

        features = self.backbone.forward_features(x)

        features = self.attention(features)

        features = self.pool(features)

        features = torch.flatten(
            features,
            1
        )

        return self.classifier(features)