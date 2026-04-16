"""
Late Fusion Module
- image encoder + tabular encoder 출력을 concat → FC → Sigmoid
- 담당: 팀원 A
"""
import torch
import torch.nn as nn
from .image_encoders import build_encoder


class MultimodalFusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fc = cfg.model.fusion

        self.image_encoder = build_encoder(cfg)
        # TODO: tabular encoder 구현 후 연결 (팀원 C)
        # self.tabular_encoder = build_tabular_encoder(cfg)

        self.classifier = nn.Sequential(
            nn.Linear(fc.concat_dim, fc.hidden_dim),
            nn.ReLU(),
            nn.Dropout(fc.dropout),
            nn.Linear(fc.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, ct, tabular):
        img_feat = self.image_encoder(ct)           # (B, img_dim)
        # tab_feat = self.tabular_encoder(tabular)  # (B, tab_dim)
        # fused = torch.cat([img_feat, tab_feat], dim=1)
        fused = img_feat                            # tabular 완성 전 임시
        return self.classifier(fused)
