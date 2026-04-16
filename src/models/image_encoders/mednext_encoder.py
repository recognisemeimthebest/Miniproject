"""
MedNeXt Encoder
- MONAI >= 1.3 의 MedNeXt에서 enc_stages + down_blocks로 encoder 추출
- pretrained: MIC-DKFZ/MedNeXt repo (수동 key 매핑 필요)
- 메모리 주의: kernel_size=3, init_filters=16 권장 (config 참고)
- 담당: 팀원 B
"""
import torch
import torch.nn as nn
from monai.networks.nets import MedNeXt


class MedNeXtEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        enc_cfg = cfg.model.image_encoder
        mednext = MedNeXt(
            in_channels=enc_cfg.in_channels,
            n_channels=enc_cfg.init_filters,
            n_classes=2,
            exp_r=2,
            kernel_size=enc_cfg.kernel_size,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=enc_cfg.blocks_down,
        )
        self.stem = mednext.stem
        self.enc_stages = mednext.enc_stages
        self.down_blocks = mednext.down_blocks

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = enc_cfg.out_dim

    def forward(self, x):
        x = self.stem(x)
        for enc, down in zip(self.enc_stages, self.down_blocks):
            x = enc(x)
            x = down(x)
        x = self.pool(x).flatten(1)           # (B, out_dim)
        return x
