from .segresnet_encoder import SegResNetEncoder
from .basicunet_encoder import BasicUNetEncoder
from .mednext_encoder import MedNeXtEncoder


def build_encoder(cfg):
    """config에서 모델명을 읽어 encoder 인스턴스 반환"""
    name = cfg.model.image_encoder.name
    encoders = {
        "SegResNetEncoder": SegResNetEncoder,
        "BasicUNetEncoder": BasicUNetEncoder,
        "MedNeXtEncoder":   MedNeXtEncoder,
    }
    if name not in encoders:
        raise ValueError(f"Unknown encoder: {name}. Choose from {list(encoders.keys())}")
    return encoders[name](cfg)
