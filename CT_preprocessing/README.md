# CT Preprocessing (M1 태스크 제출용)

NSCLC-Radiomics Lung1 데이터셋에서 **CT + GTV mask**를 CNN 학습용 고정 크기
텐서로 변환하는 전처리 코드 모음.

학습 파이프라인이 실제로 쓰는 원본은 `src/data/` 밑에 있고, 이 폴더는
팀원 리뷰/제출용으로 **동일 코드에 상세 주석**을 붙인 사본.

## 파일

- **`dataset.py`** — NSCLC-Radiomics 임상 CSV + 디스크 NIfTI를 조인해
  manifest를 만들고, stratified train/val/test split + PyTorch Dataset 제공.
  - `load_manifest()`: 임상 CSV + NIfTI 파일 존재 여부 조인, 조기 censored 제외
  - `split_manifest()`: label 기준 stratified 70/15/15 split (seed=42)
  - `NSCLCDataset`: PyTorch Dataset 클래스, MONAI transform과 연결

- **`ct_preprocess.py`** — MONAI 기반 7단계 전처리 파이프라인.
  - `CropAroundMaskCoMd`: GTV 무게중심 기준 커스텀 crop transform
    (표준 조합이 multi-blob mask에서 실패하는 22% 버그 해결)
  - `build_preprocess()`: Compose 파이프라인 생성 (train/val 공용)
  - `build_inference_preprocess()`: 추론/Streamlit용 deterministic 별칭

## 전처리 흐름

```
원본 DICOM 시리즈
    ↓  scripts/dicom_to_nifti.py
NIfTI 파일 (ct.nii.gz, gtv_mask.nii.gz)
    ↓  dataset.py: load_manifest + split_manifest
manifest DataFrame (train=293 / val=64 / test=63)
    ↓  ct_preprocess.py: build_preprocess
7단계 MONAI Compose:
  1. Load       — NIfTI → numpy + channel-first
  2. Orient     — 좌표계 RAS로 통일
  3. Resample   — 1×1×1 mm³ isotropic
  4. HU clip    — [-1000, 400] → [0, 1]
  5. KeepLargest — multi-blob mask에서 원발 GTV만
  6. CoM crop   — GTV 무게중심 기준 128×128×64
  7. Pad        — 경계 보정 zero-pad
    ↓  (training일 때 추가)
Augmentation (L-R flip, ±5° 회전, ±5% 스케일, intensity 미세 변동)
    ↓
(B, 1, 128, 128, 64) float32 ∈ [0, 1]  ← CNN 입력
```

## 설계 결정 핵심

- **HU clip 범위 `[-1000, 400]`** — Hosny et al. 2018 NSCLC deep learning
  표준값. 공기 이하/뼈 이상은 폐 병변 분석에 불필요.
- **1mm isotropic** — 환자마다 CT spacing이 달라서 그대로 쓰면 CNN이 "해상도"를
  특징으로 오인. 통일 필수.
- **KeepLargestConnectedComponent** — NSCLC-Radiomics GTV-1 ROI는 원발+림프절+위성
  blob이 섞여 있어 그냥 bbox로 감싸면 22% 환자에서 mask 실종. 원발만 유지.
- **CoM 기반 crop** — volume 중앙이 아닌 mask 무게중심에서 자름 → 항상 mask
  내부에서 crop 보장.
- **보수적 augmentation** — 의학적으로 말이 되는 변형만 (L-R flip ✅, 상하 flip ❌,
  ±5° 회전, ±5% intensity). MedicalNet pretrain 분포 유지 목적.

## 외부 의존성

- MONAI 1.3+
- PyTorch 2.0+
- pandas, scikit-learn, nibabel

## 참고

- 기획서 §2.3 (레이블 정의), §3.3.1 (전처리 상세)
- Hosny et al., *Deep learning for lung cancer prognostication*, PLoS Med 2018
- Chen et al., *MedicalNet: 3D medical image pretraining*, arXiv:1904.00625
