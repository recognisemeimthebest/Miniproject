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

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    MapTransform,
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


class CropAroundMaskCoMd(MapTransform):
    """Mask의 무게중심(Center of Mass)을 중심으로 고정 ROI 크기로 crop.

    `CropForegroundd` + `ResizeWithPadOrCropd`(volume center) 조합은 multi-blob
    mask(원발 GTV + 위성 병변 + 림프절)에서 중앙 crop window가 빈 공간에 떨어져
    mask가 통째로 사라지는 버그가 있음. 이 transform은 foreground의 무게중심을
    crop 중심으로 잡아 **항상 mask 내부에서 자르도록** 보장.

    경계에 가까우면 일부가 잘려 ROI보다 작은 volume이 나올 수 있으므로,
    뒤에 `ResizeWithPadOrCropd`를 붙여 부족분을 zero-pad 하는 것을 권장.

    Args:
        keys: crop을 적용할 key 목록 (image, mask 모두).
        source_key: 무게중심 계산에 쓸 mask key.
        roi_size: 최종 ROI 크기 (voxel).
    """

    def __init__(self, keys: Sequence[str], source_key: str, roi_size: Sequence[int]):
        super().__init__(keys)
        self.source_key = source_key
        self.roi_size = np.asarray(roi_size, dtype=int)

    def __call__(self, data):
        d = dict(data)
        mask = d[self.source_key]
        arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
        fg = arr[0] > 0 if arr.ndim == 4 else arr > 0
        vol_shape = np.asarray(fg.shape, dtype=int)

        if fg.any():
            com = np.argwhere(fg).mean(axis=0).round().astype(int)
        else:
            com = vol_shape // 2

        # CoM 주변에 roi_size만큼 잡되, 볼륨 경계 안으로 clamp
        lo = np.clip(com - self.roi_size // 2, 0, np.maximum(vol_shape - self.roi_size, 0))
        hi = np.minimum(lo + self.roi_size, vol_shape)

        for k in self.keys:
            v = d[k]
            d[k] = v[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return d


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

        # 5. Multi-blob mask(원발+림프절+위성병변) 중 가장 큰 component만 유지.
        #    NSCLC-Radiomics의 GTV-1 ROI는 공간적으로 분리된 blob이 섞여있어서
        #    합집합 bbox의 가운데가 빈 공간이 되는 경우가 많음 → 22% 환자에서
        #    mask가 통째로 사라지는 버그의 근원. 원발 GTV만 남겨 prognosis에 집중.
        KeepLargestConnectedComponentd(keys=KEYS_MASK),

        # 6. 원발 GTV 무게중심 기준으로 roi_size crop (volume 중앙이 아니라 mask 중앙).
        CropAroundMaskCoMd(keys=KEYS_BOTH, source_key=KEYS_MASK, roi_size=roi_size),

        # 7. CoM이 볼륨 경계 근처면 ROI보다 작게 나올 수 있어 부족분을 zero-pad.
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
