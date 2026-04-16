"""
SegResNet Encoder
- MONAI의 SegResNet encode() 메서드를 활용
- pretrained: MONAI Model Zoo (whole-body CT segmentation)
- 담당: 팀원 B
"""
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class SegResNetEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        enc_cfg = cfg.model.image_encoder
        self.backbone = SegResNet(
            in_channels=enc_cfg.in_channels,
            out_channels=2,                   # 분류용 dummy head, encoder만 쓸 것
            init_filters=enc_cfg.init_filters,
            blocks_down=enc_cfg.blocks_down,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = enc_cfg.out_dim

    def forward(self, x):
        # encode()는 (bottleneck, skip_list) 반환
        x, _ = self.backbone.encode(x)
        x = self.pool(x).flatten(1)          # (B, out_dim)
        return x
