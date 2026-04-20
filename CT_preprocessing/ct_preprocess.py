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
    SelectItemsd,
    Spacingd,
    ToTensord,
)

KEYS_IMAGE = "image"
KEYS_MASK = "mask"
KEYS_BOTH = [KEYS_IMAGE, KEYS_MASK]


class CropAroundMaskCoMd(MapTransform):
    """Mask의 무게중심(Center of Mass)을 중심으로 고정 ROI 크기로 crop.

    ─── 왜 이 커스텀 transform이 필요한가? ──────────────────────────────
    MONAI 표준 조합 `CropForegroundd` + `ResizeWithPadOrCropd`(volume center)는
    multi-blob mask(원발 GTV + 위성 병변 + 림프절 전이)에서 실패:
      1. CropForegroundd는 모든 blob을 감싸는 최소 bounding box를 잡음.
      2. ResizeWithPadOrCropd(volume center)는 그 bbox의 **기하학적 중심**을 자름.
      3. 그런데 bbox 중심은 blob들 사이의 **빈 공간**에 떨어지는 경우가 많음.
      4. → 결과 crop에 mask voxel이 0개인 "mask 실종" 발생.
    실제 NSCLC-Radiomics에서 이 조합으로 돌렸을 때 **22% 환자에서 mask 픽셀이 0**이
    되는 버그가 있었음. 이 transform은 foreground voxel들의 무게중심을 crop 중심으로
    잡아 **항상 mask 내부에서 자르는 것**을 보장.

    ─── 경계 케이스 ────────────────────────────────────────────────────
    종양이 볼륨 가장자리에 있으면 CoM 주변에 roi_size만큼 잡을 때 볼륨 밖으로
    넘어감 → clamp로 안으로 당기지만 그러면 roi_size보다 작은 volume이 나옴.
    그래서 호출부에선 뒤에 `ResizeWithPadOrCropd`를 붙여 부족분을 zero-pad 해야 함
    (build_preprocess의 step 7).

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
        # MONAI MetaTensor도 numpy array도 대응 (transform 파이프라인 중간 타입 호환성).
        arr = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
        # (C, H, W, D) 채널 차원이 있으면 첫 채널만 확인. 3D일 수도 있으니 방어적으로 분기.
        fg = arr[0] > 0 if arr.ndim == 4 else arr > 0
        vol_shape = np.asarray(fg.shape, dtype=int)

        if fg.any():
            # 무게중심 = foreground voxel 좌표들의 평균. 정수로 반올림 후 crop index로 사용.
            com = np.argwhere(fg).mean(axis=0).round().astype(int)
        else:
            # 방어 케이스: mask가 완전히 비어있으면 볼륨 중심으로 fallback.
            # (정상 데이터에선 이 경로 거의 안 타지만 파이프라인 크래시 방지)
            com = vol_shape // 2

        # CoM - roi_size/2 을 시작점으로 잡되:
        #  - 0 미만이면 0으로 (볼륨 밖 음수 인덱스 방지)
        #  - vol_shape - roi_size 보다 크면 당김 (끝이 볼륨 밖으로 나가지 않게)
        # np.maximum(..., 0): 볼륨이 roi_size보다 작은 변칙 케이스에서 음수 방지.
        lo = np.clip(com - self.roi_size // 2, 0, np.maximum(vol_shape - self.roi_size, 0))
        hi = np.minimum(lo + self.roi_size, vol_shape)

        # image와 mask에 **완전히 동일한 window**로 자름 → voxel-level alignment 보존.
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
    rot_deg: float = 5.0,
    scale_pct: float = 5.0,
    translate_vox: Sequence[int] = (5, 5, 3),
    intensity_scale: float = 0.05,
    intensity_shift: float = 0.05,
    affine_prob: float = 0.5,
    flip_prob: float = 0.5,
    intensity_prob: float = 0.3,
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
    # ═══════════════════════════════════════════════════════════════════
    # Deterministic pipeline (train/val/test 모두 공통) — 순서 중요!
    # 각 단계는 앞 단계 결과를 받아서 처리하므로 순서를 바꾸면 동작이 달라짐.
    # ═══════════════════════════════════════════════════════════════════
    common = [
        # ─── STEP 1: 파일 로드 ──────────────────────────────────────────
        # LoadImaged: NIfTI 파일을 읽어 numpy array + meta(affine, spacing) 로드.
        #   image_only=False → meta_dict를 같이 유지 (Spacingd가 physical spacing 정보 필요).
        # EnsureChannelFirstd: MONAI는 channel-first 포맷 가정.
        #   (H,W,D) → (1,H,W,D) 로 채널 차원 삽입.
        LoadImaged(keys=KEYS_BOTH, image_only=False),
        EnsureChannelFirstd(keys=KEYS_BOTH),

        # ─── STEP 2: 좌표계 통일 (RAS) ─────────────────────────────────
        # DICOM 원본은 환자마다 방향(axcode)이 다름: LPS, RAI, PIL 등 제각각.
        # RAS (Right-Anterior-Superior)로 통일해야:
        #   (a) L-R flip augmentation이 의미 있게 됨 (x축이 항상 좌↔우)
        #   (b) 모델이 "방향"을 우연한 특징으로 학습하지 않음
        #   (c) 추론 시 다른 시스템 데이터도 같은 orientation으로 받아들임
        Orientationd(keys=KEYS_BOTH, axcodes="RAS"),

        # ─── STEP 3: Isotropic resampling → 1×1×1 mm³ ─────────────────
        # 왜 필요한가: 환자마다 CT spacing이 다름 (in-plane 0.7~0.9mm, slice 1.5~5mm 등).
        #   그대로 쓰면 CNN이 "해상도"를 병변 특징으로 오해할 위험.
        #   1mm isotropic으로 통일해야 공간 특징의 스케일이 모든 환자 동일.
        # mode: CT는 연속값이라 bilinear(부드러운 보간), mask는 라벨이라 nearest(0/1 보존).
        Spacingd(keys=KEYS_BOTH, pixdim=pixdim, mode=("bilinear", "nearest")),

        # ─── STEP 4: HU clip + [0,1] 정규화 ───────────────────────────
        # HU (Hounsfield Unit) 값 의미: 공기=-1000, 지방=-100, 물=0, 근육=40, 뼈=+300~3000
        # [-1000, 400] 범위로 clip하는 이유:
        #   -1000 이하: 공기 (CT 기기 밖) — 실제 정보 없음
        #   +400 이상: 뼈 내부 고밀도 — 폐 병변 분석에 불필요 (계산 낭비)
        # 이 파라미터 값은 Hosny et al. 2018 (NSCLC deep learning 표준 논문) 관례.
        # clip=True: 범위 밖은 경계값으로 clamp (Windowing) 후 [0,1]로 min-max 정규화.
        # → CNN이 먹기 좋은 정규화된 입력 완성.
        ScaleIntensityRanged(
            keys=KEYS_IMAGE,
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # ─── STEP 5: Multi-blob mask 정리 (핵심 버그픽스) ─────────────
        # NSCLC-Radiomics의 GTV-1 ROI는 공간적으로 분리된 blob들이 섞여 있음:
        #   원발 종양 + 림프절 전이 + 위성 병변 (+ 때때로 석회화 등 노이즈)
        # 이걸 그냥 bbox로 감싸면 blob 사이 빈 공간 때문에 Step 6의 CoM crop이
        # 엉뚱한 곳을 자름 → 초기 구현에서 22% 환자 mask 실종 버그의 원인.
        # 해결: 가장 큰 connected component = 원발 GTV만 유지.
        # (M1 prognosis는 원발 종양 기반이라 타당. 림프절/위성은 M2+ multimodal에서.)
        KeepLargestConnectedComponentd(keys=KEYS_MASK),

        # ─── STEP 6: GTV 중심 crop (128×128×64) ──────────────────────
        # 볼륨 전체(~512³) 중 95%는 관심 영역 밖 → GTV 주변만 뽑아내야 GPU/학습 효율↑.
        # 커스텀 transform: volume 중앙이 아니라 **mask의 무게중심**을 crop 중심으로.
        # (위 CropAroundMaskCoMd 클래스 docstring 참조)
        CropAroundMaskCoMd(keys=KEYS_BOTH, source_key=KEYS_MASK, roi_size=roi_size),

        # ─── STEP 7: 경계 보정 zero-pad ──────────────────────────────
        # Step 6의 CoM이 볼륨 가장자리에 있을 경우 roi_size보다 작게 잘려 나올 수 있음.
        # 이걸 zero-pad로 채워 **항상 정확히 (128,128,64) 출력**을 보장 → batch collate 가능.
        ResizeWithPadOrCropd(keys=KEYS_BOTH, spatial_size=roi_size),
    ]

    # SelectItemsd에 넘길 whitelist: 배치에 최종적으로 포함할 키들만 유지.
    # 이유: MONAI가 붙이는 meta_dict(원본 affine, 파일 경로 등)가 기본 collate와 충돌하고,
    # PersistentDataset 디스크 캐시 용량을 수십 배 키움. → 학습에 불필요한 건 전부 버림.
    keep_keys = ["image", "mask", "label", "patient_id"]

    # ═══════════════════════════════════════════════════════════════════
    # Val/Test 경로: augmentation 없이 deterministic만.
    # ═══════════════════════════════════════════════════════════════════
    if not training:
        return Compose(common + [ToTensord(keys=KEYS_BOTH), SelectItemsd(keys=keep_keys)])

    # ═══════════════════════════════════════════════════════════════════
    # Training 경로: augmentation 추가 (보수적 세트, 기획서 §3.3.1)
    # ─── 철학: "의학적으로 말이 되는 변형만 허용" ───────────────────────
    # 의료 영상은 해부학적 정합성이 중요 → aggressive augmentation 금지.
    # 자연 이미지용 CutMix/MixUp/elastic deformation은 여기서 쓰면 안 됨.
    # ═══════════════════════════════════════════════════════════════════
    rot_rad = float(rot_deg) * np.pi / 180.0       # 도(degree) → 라디안(radian) 변환
    scale_frac = float(scale_pct) / 100.0          # 퍼센트 → 비율 (5% → 0.05)
    augment = [
        # ─── L-R flip ────────────────────────────────────────────────
        # 왜 0축(좌우)만? 좌/우 폐는 근사 대칭이라 flip해도 "환자 몸" 형태 유지.
        # 상하(Z axis) flip은 금지: 머리와 발이 바뀌면 해부학이 완전히 망가짐.
        # 앞뒤(Y axis) flip도 금지: 흉골이 등으로 가버림.
        RandFlipd(keys=KEYS_BOTH, spatial_axis=0, prob=flip_prob),

        # ─── 아핀 변형 (회전 + 스케일 + 이동) ────────────────────────
        # 모두 nnU-Net 관례 수준의 **작은** 값:
        #   ±5° 회전: 실제 환자 자세 오차 범위. 그 이상이면 흉곽 비틀림 비현실적.
        #   ±5% 스케일: 환자 체형 차이 시뮬레이션.
        #   ±5 voxel 이동: crop 위치 미세 변동 (CoM 반올림 오차 보완).
        # mask는 nearest (라벨 이진성 보존), padding_mode=zeros (밖은 공기로).
        RandAffined(
            keys=KEYS_BOTH,
            rotate_range=(rot_rad, rot_rad, rot_rad),
            scale_range=(scale_frac, scale_frac, scale_frac),
            translate_range=tuple(translate_vox),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            prob=affine_prob,
        ),

        # ─── Intensity 변동 (약하게) ────────────────────────────────
        # MedicalNet pretrain 가중치는 특정 HU 분포를 가정함 → 너무 크게 흔들면
        # pretrain feature가 망가짐. 스캐너 calibration 오차 수준(±5%)만.
        # RandScaleIntensityd: x ← x * (1 + factor)  (배수)
        # RandShiftIntensityd: x ← x + offset        (덧셈)
        RandScaleIntensityd(keys=KEYS_IMAGE, factors=intensity_scale, prob=intensity_prob),
        RandShiftIntensityd(keys=KEYS_IMAGE, offsets=intensity_shift, prob=intensity_prob),

        # 마지막: numpy → torch.Tensor 변환 + 불필요한 메타 필드 제거.
        ToTensord(keys=KEYS_BOTH),
        SelectItemsd(keys=keep_keys),
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
