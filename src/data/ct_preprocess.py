"""CT CNN 입력용 전처리 파이프라인 (MONAI 기반).

NIfTI(`ct.nii.gz`, `gtv_mask.nii.gz`) → CNN 학습/추론용 4D 텐서
  shape: (C=1, H=128, W=128, D=64), float32, 값 범위 [0, 1]

기획서 §3.3.1 파라미터 그대로:
  - HU clip [-1000, 400]
  - [0, 1] min-max 정규화
  - Isotropic resampling 1×1×1 mm³
  - GTV 중심 128×128×64 crop
  - (train) 보수적 augmentation: ±5° 회전, ±5% intensity, L-R flip

학습/검증/추론에 동일 함수 사용. Streamlit 앱에서도 재활용.
"""
from __future__ import annotations

from typing import Sequence

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

KEYS_IMAGE = "image"
KEYS_MASK = "mask"
KEYS_BOTH = [KEYS_IMAGE, KEYS_MASK]


def build_preprocess(
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    roi_size: Sequence[int] = (128, 128, 64),
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    training: bool = False,
) -> Compose:
    """전처리 파이프라인 생성.

    Args:
        hu_min, hu_max: HU clipping 범위 (Hosny 2018 / MONAI 관례 = -1000, 400)
        roi_size: 최종 출력 voxel 크기 (GTV 중심 crop 후 pad/crop)
        pixdim: isotropic resampling 목표 spacing (mm)
        training: True면 augmentation 포함, False면 deterministic only

    입력 dict 예:
        {"image": "/path/to/ct.nii.gz", "mask": "/path/to/gtv_mask.nii.gz"}

    출력 dict:
        {"image": Tensor(1, 128, 128, 64), "mask": Tensor(1, 128, 128, 64), ...}
    """
    common = [
        # 1. 파일 로드 & 채널 축 정리 (monai는 channel-first 가정)
        LoadImaged(keys=KEYS_BOTH, image_only=False),
        EnsureChannelFirstd(keys=KEYS_BOTH),

        # 2. 좌표계 통일 (RAS) — 환자마다 DICOM 방향이 달라도 통일됨
        Orientationd(keys=KEYS_BOTH, axcodes="RAS"),

        # 3. Isotropic resampling → 1×1×1 mm³
        #    CT는 bilinear, mask는 nearest (라벨 보존)
        Spacingd(keys=KEYS_BOTH, pixdim=pixdim, mode=("bilinear", "nearest")),

        # 4. HU clip + [0,1] 정규화 (한 스텝으로 처리)
        ScaleIntensityRanged(
            keys=KEYS_IMAGE,
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # 5. GTV mask 중심으로 foreground crop (빈 공간 제거)
        CropForegroundd(keys=KEYS_BOTH, source_key=KEYS_MASK, allow_smaller=True),

        # 6. 고정 크기로 맞추기 (작으면 pad, 크면 center crop)
        ResizeWithPadOrCropd(keys=KEYS_BOTH, spatial_size=roi_size),
    ]

    if not training:
        return Compose(common + [ToTensord(keys=KEYS_BOTH)])

    # ── 학습 시 augmentation (보수적 세트, 기획서 §3.3.1) ──
    augment = [
        # L-R flip만 (좌/우 폐 대칭적으로 학습 가능). 상하 flip은 해부학 망가뜨림 → 금지
        RandFlipd(keys=KEYS_BOTH, spatial_axis=0, prob=0.5),

        # 작은 회전·스케일·이동 (nnU-Net 스타일 보수값)
        RandAffined(
            keys=KEYS_BOTH,
            rotate_range=(0.087, 0.087, 0.087),   # ±5° in radians
            scale_range=(0.05, 0.05, 0.05),       # ±5%
            translate_range=(5, 5, 3),            # voxel 단위
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            prob=0.5,
        ),

        # Intensity (HU 분포가 MedicalNet pretrain과 너무 벌어지지 않게 약하게)
        RandScaleIntensityd(keys=KEYS_IMAGE, factors=0.05, prob=0.3),
        RandShiftIntensityd(keys=KEYS_IMAGE, offsets=0.05, prob=0.3),

        ToTensord(keys=KEYS_BOTH),
    ]
    return Compose(common + augment)


def build_inference_preprocess(
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    roi_size: Sequence[int] = (128, 128, 64),
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Streamlit 앱·평가용 deterministic 파이프라인 (training=False alias)."""
    return build_preprocess(
        hu_min=hu_min, hu_max=hu_max, roi_size=roi_size, pixdim=pixdim, training=False
    )


__all__ = [
    "build_preprocess",
    "build_inference_preprocess",
    "KEYS_IMAGE",
    "KEYS_MASK",
    "KEYS_BOTH",
]
