import torch
import torch.nn as nn
import timm


# ============================================================
# CHANNEL ATTENTION MODULE
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super(ChannelAttention, self).__init__()

        reduced_channels = max(
            channels // reduction,
            1
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                reduced_channels,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        attention = self.avg_pool(
            x
        )

        attention = self.fc(
            attention
        )

        return x * attention


# ============================================================
# PROPOSED ATTENTION EFFICIENTNET-B0 MODEL
# ============================================================

class AttentionEfficientNetB0(nn.Module):

    def __init__(
        self,
        num_classes=6,
        dropout=0.3,
        pretrained=True
    ):

        super(
            AttentionEfficientNetB0,
            self
        ).__init__()


        # ----------------------------------------------------
        # EFFICIENTNET-B0 BACKBONE
        # ----------------------------------------------------

        self.backbone = timm.create_model(

            "efficientnet_b0",

            pretrained=pretrained,

            num_classes=0,

            global_pool=""
        )


        # ----------------------------------------------------
        # FEATURE DIMENSION
        # ----------------------------------------------------

        feature_channels = self.backbone.num_features


        # ----------------------------------------------------
        # CHANNEL ATTENTION
        # ----------------------------------------------------

        self.channel_attention = ChannelAttention(

            channels=feature_channels

        )


        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(

            1

        )


        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.BatchNorm1d(
                feature_channels
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                feature_channels,
                num_classes
            )
        )


    def forward(self, x):

        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        x = self.backbone.forward_features(

            x

        )


        # ----------------------------------------------------
        # APPLY CHANNEL ATTENTION
        # ----------------------------------------------------

        x = self.channel_attention(

            x

        )


        # ----------------------------------------------------
        # GLOBAL POOLING
        # ----------------------------------------------------

        x = self.global_pool(

            x

        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        x = self.classifier(

            x

        )


        return x


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    model = AttentionEfficientNetB0(

        num_classes=6,

        dropout=0.3,

        pretrained=False

    )


    model = model.to(

        device

    )


    print(

        "\n" + "=" * 70

    )


    print(

        "ATTENTION-EFFICIENTNET-B0 MODEL"

    )


    print(

        "=" * 70

    )


    print(

        model

    )


    print(

        "\nModel created successfully!"

    )


    print(

        f"Device: {device}"

    )