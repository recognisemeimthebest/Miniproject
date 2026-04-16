# LUNA-XAI Mini — NSCLC 2년 생존 예측 멀티모달 XAI

> **NSCLC CT + Radiomics + Clinical 변수를 결합한 멀티모달 딥러닝 생존 예측 파이프라인**  
> 최종 프로젝트를 위한 팀 workflow 및 기술 리허설 목적의 미니 프로젝트

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터 | NSCLC-Radiomics (TCIA Lung1, n=419) |
| 타겟 | 2년 생존 이진분류 |
| 이미지 모델 | SegResNet / BasicUNet / MedNeXt (encoder 재활용) |
| Fusion | Late fusion (concat → FC → Sigmoid) |
| XAI | Grad-CAM (CT), SHAP (Radiomics+Clinical), Modality Ablation |
| 서비스 | Streamlit 로컬 데모 |

---

## 폴더 구조

```
.
├── configs/                # 모델별 실험 설정 (yaml)
│   ├── segresnet.yaml
│   ├── basicunet.yaml
│   └── mednext.yaml
├── src/
│   ├── data/               # 전처리 공통 모듈
│   ├── models/
│   │   ├── image_encoders/ # SegResNet, BasicUNet, MedNeXt encoder
│   │   ├── tabular_encoders/
│   │   └── fusion.py
│   ├── train/
│   ├── evaluate/
│   └── explain/            # Grad-CAM, SHAP, Ablation
├── experiments/            # 실험별 결과 폴더
│   ├── exp01_segresnet/
│   ├── exp02_basicunet/
│   └── exp03_mednext/
├── notebooks/              # EDA, 분석용
├── app/                    # Streamlit 데모
├── data/                   # ← .gitignore 처리 (README만 추적)
└── checkpoints/            # ← .gitignore 처리
```

---

## 빠른 시작

### 1. 환경 설치

```bash
conda env create -f environment.yml
conda activate luna-xai
```

### 2. 데이터 준비

`data/README.md` 참고 — TCIA에서 다운로드 후 지정 경로에 배치

### 3. 학습 실행

```bash
# config 파일 지정으로 모델 교체 가능
python src/train/trainer.py --config configs/segresnet.yaml
python src/train/trainer.py --config configs/basicunet.yaml
python src/train/trainer.py --config configs/mednext.yaml
```

### 4. Streamlit 데모

```bash
streamlit run app/main.py
```

---

## 브랜치 전략

```
main   ← 보호됨, PR + 리뷰 1명 필수
└── dev
    ├── feat/data-pipeline
    ├── feat/segresnet
    ├── feat/basicunet
    ├── feat/mednext
    └── feat/xai
```

---

## 팀 역할

| 역할 | 담당 폴더 |
|---|---|
| PM / 통합 | `src/models/fusion.py`, `app/` |
| 영상 파이프라인 | `src/models/image_encoders/`, `src/explain/gradcam.py` |
| Radiomics & Tabular | `src/models/tabular_encoders/`, `src/data/` |
| XAI & 평가 | `src/explain/`, `src/evaluate/` |

---

## 데이터 인용 (필수)

> Aerts et al. (2014). *Data From NSCLC-Radiomics (version 4)*. TCIA.  
> https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI  
> License: CC BY-NC 3.0

---

*Python 3.10 / PyTorch 2.x / CUDA 12 / MONAI ≥ 1.3*
