"""
BasicUNet Encoder
- MONAI BasicUNet의 encoder stage만 수동으로 forward
- pretrained weight 없음 → from scratch
- 담당: 팀원 B
"""
import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet


class BasicUNetEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        enc_cfg = cfg.model.image_encoder
        unet = BasicUNet(
            spatial_dims=3,
            in_channels=enc_cfg.in_channels,
            out_channels=2,
            features=enc_cfg.features,
        )
        # encoder stage만 추출
        self.enc0 = unet.conv_0
        self.enc1 = unet.down_1
        self.enc2 = unet.down_2
        self.enc3 = unet.down_3
        self.enc4 = unet.down_4

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = enc_cfg.out_dim        # features[-2] = 256

    def forward(self, x):
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.pool(x).flatten(1)           # (B, out_dim)
        return x
