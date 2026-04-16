# NSCLC CT 기반 멀티모달 2년 생존 예측 및 설명가능 AI 프로젝트 기획서

**프로젝트명(가제):** *LUNA-XAI (LUng NSCLC Analysis with eXplainable AI)*
**기간:** 10일 (Day 1 ~ Day 10)
**인원:** 4명
**목적:** 내부 스터디 — 멀티모달 딥러닝 + 생존 예측 + XAI 파이프라인을 팀 전체가 end-to-end로 경험하고 최종 발표자료로 정리

---

## 1. 프로젝트 개요

### 1.1 배경 및 동기

비소세포폐암(NSCLC, Non-Small Cell Lung Cancer)은 전체 폐암의 약 85%를 차지하며, 1차 치료 후 2년 시점의 생존 여부는 예후 판정과 치료 계획에서 핵심 지표로 쓰인다. 단일 영상 소견이나 단일 임상 변수만으로 예측하는 기존 접근은 한계가 명확하며, 최근에는 **CT + 정량적 영상 특징(radiomics) + 임상 변수**를 결합한 멀티모달 딥러닝 접근이 성능과 임상 해석력 측면에서 우위를 보이고 있다. 동시에 **"왜 그렇게 예측했는지"**를 설명할 수 있는 XAI는 임상 수용성 확보의 전제 조건이다.

본 스터디는 공개 데이터셋(NSCLC-Radiomics, TCIA)을 이용해 NIH-MIP의 `Multimodal_RPModel` 아키텍처 패턴을 **NSCLC 도메인에 맞게 이식**하고, 여기에 `cgiova/multimodal-xai-prostate`의 설명 방식을 **데이터 특성에 맞게 재구현**하는 것을 목표로 한다.

### 1.2 프로젝트 질문

> **"NSCLC 환자의 pre-treatment CT 영상 + PyRadiomics 피처 + 임상 변수를 통합한 멀티모달 모델이, 단일 모달리티 대비 2년 생존 예측에서 얼마나 개선되며, 그 예측 근거는 임상적으로 타당한가?"**

### 1.3 목표 (SMART)

| 구분 | 목표 |
|---|---|
| 주요 목표 | 2년 생존 예측 이진분류에서 멀티모달 모델(M4) AUROC ≥ 0.75 달성 |
| 부수 목표 | 단일 모달리티(M1/M2/M3) 대비 AUROC 유의한 개선 (DeLong test p<0.05) |
| 해석 목표 | Grad-CAM 히트맵이 GTV-1 영역과 중첩되는지 정성·정량 평가, SHAP 상위 피처가 문헌상 예후인자와 일치하는지 검증 |
| 학습 목표 | 팀원 4명 전원이 모달리티별 학습 → 멀티모달 fusion → XAI 적용의 전 과정을 설명·재현 가능 |

---

## 2. 데이터

### 2.1 데이터셋: NSCLC-Radiomics (TCIA Lung1)

- **출처:** The Cancer Imaging Archive, DOI: 10.7937/K9/TCIA.2015.PF0M9REI
- **라이선스:** CC BY-NC 3.0 (비상업적 연구 목적 허용, 인용 필수)
- **원본 규모:** 422명 NSCLC 환자, 사전 치료(pre-treatment) CT + 수동 delineation + 임상 outcome
- **본 프로젝트 사용 n:** **419명** (GTV mask 이상·불완전 volume 등 3명 제외)
- **데이터 준비 상태:** 다운로드·DICOM→NIfTI 변환·GTV mask 추출·Radiomics 피처 추출 **완료** (프로젝트 시작 시점 기준)

### 2.2 데이터 구성

| 데이터 유형 | 포맷 | 설명 |
|---|---|---|
| CT 영상 | DICOM → NIfTI (전처리 완료) | 축방향 slice **512×512**, 환자별 slice 수 상이 |
| 종양 segmentation | SEG/RTSTRUCT → binary mask | GTV-1 mask (primary lung lesion) |
| 임상 데이터 | CSV (Lung1.clinical) | age, clinical.T/N/M.Stage, Overall.Stage, Histology, gender, Survival.time, deadstatus.event |
| Radiomics 피처 | CSV (추출 완료) | IBSI-compliant hand-crafted features |

### 2.3 타겟 정의

- **2년 생존 이진분류:**
  - `survival.time ≥ 730 days AND deadstatus.event == 0` → **Class 1 (Survivor)**
  - `survival.time < 730 days AND deadstatus.event == 1` → **Class 0 (Non-survivor)**
  - `survival.time < 730 days AND deadstatus.event == 0` (right-censored) → **제외**
- 실제 분석 n은 419명 중 censoring 제외 후 결정됨 (EDA 시점에 확정)

### 2.4 데이터 분할 전략

- Train : Validation : Test = **60 : 20 : 20** (stratified by label × Overall Stage)
- 재현성을 위한 random seed 고정 (seed=42), split indices CSV로 팀 공유
- Test set은 프로젝트 후반부까지 **blind** 유지 (data leakage 방지)

---

## 3. 방법론

### 3.1 전체 파이프라인 아키텍처

```
┌────────────────────────────────────────────────────────────────────────┐
│                 NSCLC-Radiomics (n=419, 전처리 완료)                   │
└────────────────────────────────────────────────────────────────────────┘
          │                    │                      │
          ▼                    ▼                      ▼
    ┌──────────┐        ┌────────────┐        ┌────────────┐
    │   CT     │        │  Radiomics │        │  Clinical  │
    │  NIfTI   │        │   (CSV,    │        │  (CSV)     │
    │ 512×512  │        │  추출완료) │        │            │
    └────┬─────┘        └──────┬─────┘        └──────┬─────┘
         │                     │                     │
         ▼                     ▼                     ▼
    Input adaptation      Feature selection     Tabular encoding
    (ROI crop/resize,     (LASSO, corr filter,  (one-hot, scaling,
     HU clip, normalize)   train-set only)       missing imputation)
         │                     │                     │
         ▼                     ▼                     ▼
    ┌──────────┐        ┌────────────┐        ┌────────────┐
    │  3D CNN  │        │ Tabular    │        │ MLP        │
    │ (ResNet- │        │ Branch     │        │ encoder    │
    │ 10 3D,   │        │ (MLP)      │        │            │
    │ MedNet   │        │            │        │            │
    │ init)    │        │            │        │            │
    └────┬─────┘        └──────┬─────┘        └──────┬─────┘
         │                     │                     │
         └─────────────┬───────┴─────────────┬───────┘
                       │                     │
                       ▼                     ▼
                  ┌────────────────────────────────┐
                  │    Late Fusion (Concat)        │
                  │    → FC → Sigmoid              │
                  └────────────┬───────────────────┘
                               │
                               ▼
                    P(2-year survival)
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
      Grad-CAM             SHAP                  Modality
      (CT branch)      (Radiomics + Clinical)    Ablation
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │   Streamlit 로컬 데모 (서비스)    │
              │   ──────────────────────────     │
              │   [입력] 임상 변수 폼            │
              │        + CT NIfTI 업로드         │
              │   [추론] M4 예측 P(생존)         │
              │   [설명] Grad-CAM · SHAP ·       │
              │          Modality Ablation       │
              └──────────────────────────────────┘
```

### 3.2 예측 모델 (NIH-MIP `Multimodal_RPModel` 패턴 이식)

NIH-MIP 원본 연구(Simon et al., *Clinical Imaging* 2025)는 4개 모델을 비교하는 구조를 가진다. 본 프로젝트는 이 패턴을 그대로 채용한다.

| 모델 ID | 입력 모달리티 | 목적 |
|---|---|---|
| **M1** | Clinical only (age, stage, histology, gender) | 임상 단독 baseline |
| **M2** | CT only (3D CNN) | 영상 단독 성능 |
| **M3** | Radiomics + Clinical | 핸드크래프트 피처 기반 멀티모달 |
| **M4** | CT + Radiomics + Clinical (**fully automated multimodal**) | 최종 제안 모델 |

Baseline 비교군으로 **TNM Stage 단독 logistic regression** 모델을 함께 평가한다.

### 3.3 각 모달리티 처리 방법

#### 3.3.1 CT 이미지 (CNN branch)

**원본 데이터:** 축방향 slice **512×512**, voxel spacing은 환자별로 다를 수 있음.

- **입력 adaptation (이미 전처리된 NIfTI에서 출발):**
  - HU clipping: [-1000, 400] → 정규화 [0, 1]
  - GTV-1 중심 3D ROI crop: **128×128×64 voxels** (네트워크 입력 크기)
  - 원본 512×512 slice를 그대로 쓰지 않는 이유: 대부분의 voxel이 폐·배경이라 학습 비효율, GPU 메모리 과다
  - 필요 시 isotropic resampling (1×1×1 mm³)
- **모델:** 3D ResNet-10, **MedicalNet pretrained weight** 로드 (전이학습)
- **Augmentation:** random flip, rotation (±10°), intensity shift (±5%)
- **GPU 메모리 전략:** mixed precision (AMP), batch size 4, gradient checkpointing (필요 시)

#### 3.3.2 Radiomics (tabular branch — 추출 완료)

- **추출 완료 가정** — 프로젝트는 추출된 CSV에서 시작
- **피처군 (IBSI-compliant, PyRadiomics 기반):**
  - Shape, First-order, GLCM, GLRLM, GLSZM, NGTDM, GLDM
  - 총 약 100개 피처 (wavelet 제외 기준)
- **피처 선택 (본 프로젝트에서 수행):**
  - 상수 피처 / 고상관(|r|>0.95) 피처 제거
  - LASSO로 차원 축소 → 상위 20~30개 유지
  - **중요:** 모든 selection은 **train set 기준으로만** 수행, validation/test는 같은 피처 세트만 사용
- **정규화:** Z-score standardization (train set 평균/표준편차로 val/test 적용)

#### 3.3.3 임상 변수 (tabular branch)

- Age (연속형, z-score)
- Clinical T/N/M stage (ordinal encoding)
- Overall Stage (one-hot)
- Histology (one-hot: adenocarcinoma, large-cell carcinoma, squamous-cell carcinoma, NOS)
- Gender (binary)
- 결측값은 train set의 중앙값/최빈값 대체, 결측 flag 피처 추가

### 3.4 Fusion 전략

- **Late Fusion (concatenation):** 각 branch의 feature embedding을 concat 후 FC layer 통과
  - CT branch output: 512-dim
  - Radiomics MLP output: 64-dim
  - Clinical MLP output: 32-dim
  - → concat (608-dim) → FC(128) → Dropout(0.3) → FC(1) → Sigmoid
- **대안 실험 (시간 여유 시):** attention-based fusion, gating mechanism

### 3.5 학습 설정

| 항목 | 값 |
|---|---|
| Loss | Binary Cross-Entropy (with class weight for imbalance) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR |
| Batch size | 4 (3D CNN), 32 (tabular-only) |
| Epochs | 50 (early stopping patience=10 on val AUROC) |
| Cross-validation | Train set 내 5-fold stratified CV로 하이퍼파라미터 튜닝 |
| 재현성 | seed=42, `torch.use_deterministic_algorithms(True)` |

### 3.6 평가 지표

- **주 지표:** AUROC, Sensitivity, Specificity, F1
- **보조 지표:** Balanced Accuracy, AUPRC (class imbalance 고려)
- **통계 검정:** DeLong test로 모델 간 AUROC 차이 유의성 검증
- **보정(Calibration):** Brier score, calibration plot
- **생존 관점 보조 분석:** 예측 확률로 high/low risk 이분화 후 Kaplan-Meier + log-rank test

### 3.7 XAI 재구현 전략 (`cgiova/multimodal-xai-prostate` 원리 차용, 데이터 맞춤 재구현)

원본 레포가 전립선 MRI 특화이므로 **동일 원리**(이미지-공간 설명 + tabular-피처 설명 + 모달리티 기여도)를 유지하되 **구현은 NSCLC/CT에 맞게 새로 작성**한다.

#### 3.7.1 이미지 설명 (CT branch)
- **Grad-CAM / Grad-CAM++** (`pytorch-grad-cam` 라이브러리)
  - 마지막 3D conv layer의 attention map을 3D volume으로 시각화
  - 축방향 최대 강도 투영(MIP) 2D 오버레이로 리포팅 (발표자료 수록)
- **정량 평가:**
  - **IoU(Grad-CAM heatmap binarized vs. GTV-1 mask)** — 설명의 해부학적 타당성 측정
  - **Pointing game:** heatmap peak voxel이 GTV-1 내부에 위치하는지의 비율
  - 정량 지표는 test set 전수에 대해 계산, 평균 ± 표준편차 제시

#### 3.7.2 Tabular 설명 (Radiomics + Clinical)
- **SHAP** (`shap` 라이브러리)
  - Global: summary plot (beeswarm), bar plot feature importance
  - Local: 개별 환자 force plot / waterfall
- **비교 검증:** SHAP 상위 피처가 NSCLC 문헌상 예후 인자(Overall stage, 종양 volume, entropy-기반 texture feature 등)와 일치하는지 정성 검토

#### 3.7.3 모달리티 기여도 분석 (Multimodal-specific)
- **Modality ablation:** 학습된 M4에서 각 모달리티 입력을 mean/zero로 masking → AUROC 변화량 측정
- **Modality-level SHAP (KernelExplainer):** 3개 모달리티 embedding을 "super-feature"로 간주해 SHAP 값 계산

#### 3.7.4 설명 품질 평가
- **Faithfulness:** SHAP 상위 k개 피처 제거 시 성능 하락 폭 (deletion metric)
- **Stability:** 유사 입력에 대한 설명 일관성 (cosine similarity of attribution vectors)
- **Plausibility:** 임상 지식과의 부합도 — 내부 정성 평가 + 팀 내 의료 배경 보유자(간호/의료AI) 리뷰

> **일정 제약에 따른 trade-off:** 서비스 구현을 10일 내에 포함하기 위해 Faithfulness/Stability는 **Stretch(선택) 항목으로 이동**. 필수 정량평가는 IoU + Pointing game으로 한정.

### 3.8 서비스 구현 (Streamlit 로컬 데모)

학습·평가·XAI 완료 후, M4 모델을 실시간 추론 가능한 **Streamlit 웹 앱**으로 포장한다. 발표 당일 로컬에서 실행하고 화면 공유 방식으로 시연한다. **배포는 로컬에 한정**, 외부 호스팅 없음.

#### 3.8.1 서비스 아키텍처

```
┌────────────────────── Streamlit App ──────────────────────┐
│                                                            │
│  [Sidebar]                  [Main Area]                    │
│  ─────────                  ─────────────                  │
│   Step 1. 임상 변수 입력     [입력 요약 카드]                │
│    ├ Age (slider)           ▼                              │
│    ├ T/N/M stage (select)   [추론 버튼 "Predict"]          │
│    ├ Overall Stage          ▼                              │
│    ├ Histology              ┌─────────────────────────┐   │
│    └ Gender                 │  Output Panel (tabs)    │   │
│                             │  ─────────────────────  │   │
│   Step 2. CT 업로드         │  [Tab 1] 예측 결과       │   │
│    ├ .nii.gz / .nii         │   • P(2-yr survival)    │   │
│    └ GTV mask (.nii)        │   • Risk category       │   │
│                             │   • Confidence bar      │   │
│   [Run Inference 버튼]       │                         │   │
│                             │  [Tab 2] Grad-CAM       │   │
│                             │   • CT + heatmap overlay│   │
│                             │   • 3-plane view        │   │
│                             │                         │   │
│                             │  [Tab 3] SHAP           │   │
│                             │   • Waterfall plot      │   │
│                             │   • Top-10 features     │   │
│                             │                         │   │
│                             │  [Tab 4] Modality       │   │
│                             │         Ablation        │   │
│                             │   • ΔAUROC bar chart    │   │
│                             │   • Contribution pie    │   │
│                             └─────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

#### 3.8.2 입력 화면 (Input Panel)

**Sidebar Step 1: 임상 변수 입력**

| 변수 | 위젯 | 설정 |
|---|---|---|
| Age | `st.slider` | 30 ~ 90, default 65 |
| Clinical T stage | `st.selectbox` | T1 / T2 / T3 / T4 |
| Clinical N stage | `st.selectbox` | N0 / N1 / N2 / N3 |
| Clinical M stage | `st.selectbox` | M0 / M1 |
| Overall Stage | `st.selectbox` | I / II / IIIA / IIIB / IV |
| Histology | `st.selectbox` | adenocarcinoma / large-cell / squamous-cell / NOS |
| Gender | `st.radio` | male / female |

**Sidebar Step 2: CT 업로드**
- `st.file_uploader`: NIfTI 파일(`.nii`, `.nii.gz`) 1개
- (선택) GTV mask NIfTI 파일 1개 — 없으면 예측만 가능, Grad-CAM IoU 계산 불가
- 업로드 후 middle slice 썸네일 미리보기

#### 3.8.3 추론 파이프라인

1. **입력 검증**: 필수 필드 누락 시 오류 메시지
2. **CT 전처리**: 학습 때와 동일한 전처리 재사용 (`data/ct_transforms.py`, `preprocess_clinical.py`)
3. **Radiomics 추출 (선택)**: 업로드된 CT + GTV mask가 있으면 PyRadiomics로 실시간 추출 (시간 소요 시 사전 추출 CSV에서 샘플 선택 모드 병행 제공)
4. **M4 추론**: 캐시된 체크포인트(`ckpt/m4_best.pt`) 로드, GPU/CPU 자동 감지
5. **XAI 생성**: Grad-CAM + SHAP + Ablation 병렬 계산
6. **결과 렌더**: tabs로 구분해 순차 표시

**성능 타겟:** 단일 환자 추론 + XAI 전체 ≤ **30초** (GPU 사용 시)

#### 3.8.4 출력 화면 (Output Panel — 4-Tab 구조)

**Tab 1: 예측 결과**
- 큰 숫자로 2년 생존 확률 표시 (`st.metric`)
- Risk category (Low/High) 배지, cutoff는 학습 시 도출한 threshold
- 예측 신뢰도 bar (softmax 확률 gap 기반)

**Tab 2: Grad-CAM 시각화**
- CT slice 3개 방향(axial / coronal / sagittal)에 heatmap 오버레이
- 슬라이스 번호 `st.slider`로 스크롤
- GTV mask가 업로드된 경우 IoU 수치 동반 표시
- 다운로드 버튼(PNG)

**Tab 3: SHAP 시각화**
- Waterfall plot (해당 환자 단일)
- Top-10 feature bar chart
- 각 피처의 현재 값 + SHAP contribution 표

**Tab 4: Modality Ablation**
- 3-모달 각각을 masking했을 때의 ΔP(survival) bar chart
- Modality-level SHAP: CT / Radiomics / Clinical 기여도 pie chart
- 1~2줄 자연어 해설 (예: "이 환자의 예측은 CT 영상에 주로 의존")

#### 3.8.5 구현 스택

| 구성 요소 | 기술 |
|---|---|
| UI 프레임워크 | **Streamlit** (≥1.30) |
| 모델 로딩 | PyTorch + `@st.cache_resource` (체크포인트 1회만 로드) |
| CT 시각화 | `matplotlib` + `nibabel` 기반 3-plane viewer |
| SHAP | `shap.plots.waterfall`, `force_plot` |
| 실행 방식 | `streamlit run app/main.py` → 브라우저에서 `localhost:8501` |
| 배포 | **로컬 전용**, 발표 당일 종완님 워크스테이션에서 실행 |

#### 3.8.6 파일 구조

```
app/
├── main.py              # Streamlit 엔트리포인트
├── pages/
│   ├── 1_input.py       # 입력 페이지 (변수·업로드)
│   └── 2_result.py      # 결과 페이지 (4 tabs)
├── inference/
│   ├── preprocess.py    # 학습 파이프라인 재사용
│   ├── predict.py       # M4 forward pass wrapper
│   └── xai.py           # Grad-CAM/SHAP/Ablation 통합 API
├── ui/
│   ├── sidebar.py       # 입력 위젯 모듈
│   └── plots.py         # 시각화 헬퍼
└── assets/
    └── sample_cases/    # 데모용 테스트 환자 2~3명 사전 배치
```

#### 3.8.7 데모 시나리오 (발표 당일)

1. **준비 단계:** Day 10 오전에 사전 로드 스모크 테스트, 샘플 케이스 2~3명 `app/assets/sample_cases/`에 배치
2. **발표 중 시연 흐름 (3~4분):**
   - 샘플 환자 A (low-risk 예상): 업로드 → 예측 → Grad-CAM으로 종양 영역 확인 → SHAP에서 low T stage 강조
   - 샘플 환자 B (high-risk 예상): 동일 흐름 → Modality Ablation에서 어떤 모달이 기여 컸는지 비교
3. **Fallback:** 라이브 시연 실패 대비 **사전 녹화 영상(30초)** 준비, 슬라이드에 임베드
4. **제약 고지:** 이것은 **연구용 데모**이며 임상 진단 도구가 아님을 화면 footer와 발표 초반에 명시

---

## 4. 팀 구성 및 역할 분담

팀원 4명 전원이 ML 경험 보유, 그중 2명 이상이 DICOM/의료영상에 익숙한 점을 반영해 **병렬 처리 + 주기적 통합** 구조로 설계.

| 역할 | 담당자 | 주 책임 | 보조 책임 |
|---|---|---|---|
| **PM / 통합 리드** | 팀원 A | 일정 관리, 회의 진행, 데이터 split 관리, 최종 통합·발표, **Streamlit 앱 통합** | M4 fusion 구현 |
| **영상 파이프라인 리드** (DICOM 경험) | 팀원 B | GTV ROI crop, 3D CNN(M2) 학습, Grad-CAM, IoU/Pointing game, **앱 Grad-CAM 탭** | CT QC |
| **Radiomics & Tabular 리드** (DICOM 경험) | 팀원 C | Radiomics 피처 선택, M1/M3 학습, SHAP, 임상변수 전처리, **앱 입력 폼·SHAP 탭** | Calibration |
| **XAI & 평가 리드** | 팀원 D | 멀티모달 XAI 재구현, 평가 지표 구현, 통계 검정, 시각화, **앱 Ablation 탭** | 발표자료 디자인 |

### 4.1 GPU 자원 및 분담 전략

- **하드웨어:** **RTX 4070 Ti Super (16GB VRAM, 32GB RAM) — 단독 사용 (팀 공용 1대)**
- **공유 방식:** GPU는 1대이므로 **시간 분할 공유** 필요
  - M2 (3D CNN) 학습: 주간 시간대에 담당자 독점 사용
  - M1, M3 (tabular) 학습: CPU로도 가능하거나 야간 배치로 양보
  - M4 학습: 가장 priority 높음, Phase 2 초반에 blocking 확보
- **실험 추적:** 모든 run은 wandb(또는 MLflow)에 로깅하여 오프라인에서도 팀원 모두 결과 확인 가능
- **체크포인트 정책:** epoch마다 best 모델 저장, 중단 시 재시작 가능하도록 `resume_from_checkpoint` 구현

### 4.2 협업 인프라

- **코드:** GitHub private repo, `main`/`dev`/feature branch 전략, PR 리뷰 필수
- **실험 추적:** Weights & Biases (무료 tier 충분)
- **데이터/모델 저장소:** 전처리된 NIfTI는 워크스테이션 로컬, 체크포인트·결과는 Google Drive/NAS로 백업
- **커뮤니케이션:** Slack/Discord, Daily 15분 stand-up (원격 포함)
- **문서:** Notion 또는 공유 Google Docs
- **환경 통일:** `environment.yml` (conda), `requirements.txt` 공용, Python 3.10 / PyTorch 2.x / CUDA 12

---

## 5. 일정 (10일)

데이터 준비(다운로드·DICOM 변환·GTV 추출·Radiomics 추출)가 이미 완료되어 있다는 전제로, 모델링·XAI·서비스 구현에 집중한 일정.

**날짜는 팀 논의 후 확정 — 아래는 Day 1~10의 내부 진행 순서**

| Phase | Day | 주요 마일스톤 | 담당 |
|---|---|---|---|
| **Phase 1 — Setup & Single Modality** | | | |
| | Day 1 | Kick-off, 환경 셋업(conda/wandb), EDA, 타겟 라벨 생성, Train/Val/Test split 확정 | 전원 |
| | Day 2 | Radiomics 피처 선택 (LASSO), 상관 제거, feature set 고정 → M1 학습 | C |
| | Day 3 | CT input adaptation (ROI crop, HU clip, normalize) + M2 (3D CNN, MedNet init) 학습 시작 | B |
| | Day 4 | M2 학습 완료 / 튜닝, M3 (Radiomics+Clinical) 학습 | B, C |
| | Day 5 | **중간 체크포인트 #1** — M1/M2/M3 결과 공유, 문제 해결 | 전원 |
| **Phase 2 — Multimodal & XAI** | | | |
| | Day 6 | **M4 late-fusion 구현 & 학습 시작**, 하이퍼파라미터 튜닝 | A, B, C |
| | Day 7 | M4 학습 완료, Test set 평가, DeLong test, Calibration | A, D |
| | Day 8 | Grad-CAM (CT), SHAP (Radiomics+Clinical), Modality ablation 수행, XAI 정량 평가(IoU/Pointing game) | B, C, D |
| **Phase 3 — 서비스 구현 & 발표** | | | |
| | Day 9 | **Streamlit 앱 개발** — 입력 폼 + 추론 파이프라인 + XAI 3-tab 시각화, KM 생존곡선, 임상 해석 | A, B, C, D |
| | Day 10 | 앱 통합 테스트 + 샘플 케이스 검증, 발표자료 통합, 리허설, 최종 발표 + 라이브 데모 | 전원 |

### 5.1 Day 별 상세 — 서비스 구현 세부 분해

기존 Day 9(XAI 정량 평가)를 Day 8로 통합하고, Day 9 전체를 서비스 구현에 할당했습니다. Day 10에는 앱 디버깅, 발표자료, 리허설이 병행됩니다.

#### Day 8 — XAI 생성 + 정량평가 통합 (기존 Day 8+Day 9 압축)

| 시간대 | 작업 | 담당 | 산출물 |
|---|---|---|---|
| 오전 | Grad-CAM 3D 생성 + MIP 오버레이 | B | `results/gradcam/`, `figures/gradcam_overlays/` |
| 오전 | SHAP KernelExplainer + Summary/Waterfall | C | `results/shap_values.npy`, `figures/shap_*.png` |
| 오후 | Modality Ablation + Modality-level SHAP | D | `results/ablation.csv`, `figures/modality_shap.png` |
| 오후 | IoU + Pointing game 정량평가 | B | `results/gradcam_iou.csv`, `results/pointing_game.csv` |
| 저녁 | 결과 통합 + 임상 해석 초안 | A, C | `report/shap_interpretation.md` |

> **drop 항목:** Faithfulness, Stability는 Stretch로 이동. Day 9 서비스 구현에 시간 확보.

#### Day 9 — Streamlit 앱 개발 (신규)

| 시간대 | 작업 | 담당 | 산출물 |
|---|---|---|---|
| 오전 | 앱 스켈레톤 + Sidebar 입력 폼 구현 | C | `app/main.py`, `app/ui/sidebar.py` |
| 오전 | 추론 파이프라인 래핑 (전처리→M4→XAI) | A | `app/inference/predict.py`, `xai.py` |
| 오후 | Grad-CAM 3-plane viewer 탭 | B | `app/ui/plots.py` 내 gradcam 섹션 |
| 오후 | SHAP waterfall 탭 + 입력 표시 | C | SHAP 탭 완성 |
| 오후 | Modality Ablation 탭 + pie chart | D | Ablation 탭 완성 |
| 저녁 | 전체 통합 + 샘플 케이스 2~3명 동작 확인 | A (전원) | `app/assets/sample_cases/` |

#### Day 10 — 앱 검증 + 발표자료 + 리허설 + 발표

| 시간대 | 작업 | 담당 | 산출물 |
|---|---|---|---|
| 오전 (1) | 앱 smoke test (샘플 케이스 2~3명 전수 시연) | A, B | 시연 영상 녹화 (fallback용) |
| 오전 (2) | 발표 슬라이드 통합 (모델·XAI·서비스 섹션) | 담당별 병렬 | 최종 pptx |
| 오후 (1) | 내부 리허설 (앱 라이브 시연 포함 25분) | 전원 | 피드백 메모 |
| 오후 (2) | 피드백 반영, 앱 마이너 버그 수정 | 담당별 | 최종본 |
| 저녁 | **최종 발표 (라이브 데모 + 슬라이드)** | 전원 | 발표 수행, 회고 |

### 5.2 일정 압박 대응 전략

- **Day 5 체크포인트**: M1/M2/M3 중 하나라도 수렴 실패면 즉시 scope 조정
- **Day 8 체크포인트 (신설)**: 저녁에 XAI 3종 결과 확보 여부 확인. 미확보 시 Day 9 서비스 구현에 영향 → 팀원 1명을 XAI 마무리에 할당
- **Day 9 체크포인트**: 앱 기본 골격이 안 되면 **사전 녹화 영상으로 대체**하고 라이브 데모 포기
- **서비스 기능 축소 순서 (위부터 drop)**:
  1. Modality Ablation 탭 → 정적 PNG로 대체
  2. SHAP 탭 → Summary plot 1장 이미지로 대체
  3. CT 업로드 기능 → 사전 로드 샘플 3명 드롭다운 선택만 제공
  4. 위 3개가 모두 실패해도 Grad-CAM + 예측 결과만은 반드시 유지

### 5.3 진행률 가시성

- Daily stand-up에서 각자 "어제/오늘/장애물" 1분씩 공유
- Day 5 / Day 8 체크포인트 후 traffic-light(녹/황/적) 공유
- 적색 전환 시 Stretch(Faithfulness/Stability/Attention fusion) 전부 drop, 서비스는 최소 기능으로 축소

---

## 6. 리스크 관리

| 리스크 | 가능성 | 영향도 | 대응 전략 |
|---|---|---|---|
| 3D CT 학습 GPU 메모리 부족 (16GB) | 중 | 높음 | Mixed precision, batch=2, ROI crop 축소 (96×96×48), gradient checkpointing |
| 10일 내 3D CNN 수렴 실패 | 중 | 높음 | **MedicalNet pretrained weight 필수 사용**, 안 되면 2.5D(axial 3-slice) 전환 |
| 데이터 leakage (feature selection을 전체 데이터로) | 중 | 치명적 | train set만으로 LASSO·정규화, Test는 마지막까지 격리, 코드 리뷰 |
| Class imbalance | 높음 | 중 | class weight, focal loss, 또는 오버샘플링(SMOTE, tabular only) |
| Censored data 처리 실수 | 중 | 중 | 2년 미만 사망자만 Class 0, 2년 미만 censored는 **제외** — 코드 리뷰 필수 |
| GPU 1대 시간 경합 | 높음 | 중 | 학습 스케줄 표 작성, 야간 배치 활용, tabular branch는 CPU로 분산 |
| 팀원 일정 충돌 | 중 | 중 | Daily stand-up으로 조기 발견, Day 5 체크포인트에서 scope 조정 |
| Grad-CAM이 GTV 외 영역에 반응 | 중 | 중 | lung mask로 배경 masking 후 재학습, 혹은 IoU 낮은 case를 failure case로 분석에 포함 |
| 10일 일정 지연 (버퍼 없음) | 높음 | 높음 | Stretch 항목 즉시 drop, XAI 정량 평가 중 선택적 항목 생략, Minimum 기준 우선 달성 |
| Day 9 Streamlit 앱 개발 지연 | 중 | 중 | 기능 축소 순서표(5.2)에 따라 단계적 drop, 최악 시 정적 PNG 대시보드로 대체 |
| 발표 중 라이브 데모 실패 (업로드/추론 에러) | 중 | 중 | 사전 녹화 영상 준비(Day 10 오전), 샘플 케이스 드롭다운 fallback 경로 확보 |
| 앱 추론 속도 느림 (>30s) | 중 | 낮음 | 체크포인트 `@st.cache_resource`, 샘플 사전계산 결과 미리 저장 |

---

## 7. 최종 산출물 (내부 스터디 발표용)

### 7.1 발표 자료 (핵심 deliverable)

1. **프로젝트 개요 슬라이드** (1~2장): 문제 정의, 데이터(n=419), 목표
2. **데이터 EDA** (2~3장): 코호트 특성, 생존 분포, stage 분포, censoring 처리 후 유효 n
3. **방법론** (3~4장): 전체 파이프라인 다이어그램, 각 모달리티 전처리, M1~M4 구조
4. **결과** (4~5장):
   - M1~M4 성능 비교 테이블 (AUROC, Sens, Spec, F1)
   - ROC curve (4개 모델 오버레이)
   - DeLong p-value 매트릭스
   - Kaplan-Meier 생존곡선 (M4 기반 risk stratification)
5. **XAI 분석** (3~4장):
   - Grad-CAM 예시 3~5명 (성공 1~2명 + 실패 1명) + IoU/Pointing game 수치
   - SHAP summary plot + 상위 10개 피처 설명
   - 모달리티 기여도 비교 (ablation 결과)
6. **서비스 데모** (2~3장 + 라이브 시연 3~4분):
   - 앱 아키텍처·UI 스크린샷
   - **라이브 데모:** 샘플 환자 A(low-risk) → B(high-risk) 순차 시연
   - Fallback 영상(30초) 슬라이드에 임베드
7. **논의 및 한계** (1~2장): 개선 방향, 임상 적용 시 고려사항
8. **배운 점 / 팀 회고** (1장): 스터디 목적에 부합하는 learning 정리

### 7.2 부속 산출물

- GitHub repo (재현 가능한 코드 + README + 환경설정)
- `app/` 디렉토리: Streamlit 앱 소스 + 샘플 케이스 + 실행 가이드
- `results/` 폴더: 평가 지표 CSV, 그림 PNG/PDF
- 모델 체크포인트 (Git LFS 또는 외부 저장소)
- `REPORT.md`: 발표 내용의 텍스트 버전
- 라이브 데모 fallback용 녹화 영상 (30초~1분)

---

## 8. 참고 자료

### 8.1 주요 참조 레포지토리

| 용도 | 링크 |
|---|---|
| 예측 모델 패턴 | [NIH-MIP/Multimodal_RPModel](https://github.com/NIH-MIP/Multimodal_RPModel) |
| XAI 원리 | [cgiova/multimodal-xai-prostate](https://github.com/cgiova/multimodal-xai-prostate) |
| 데이터셋 | [TCIA NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) |
| 3D CNN pretrained | [MedicalNet (Tencent)](https://github.com/Tencent/MedicalNet) |
| Radiomics 추출 | [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) |
| Grad-CAM | [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) |
| SHAP | [slundberg/shap](https://github.com/shap/shap) |

### 8.2 핵심 논문

- Aerts HJWL, et al. *Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach.* **Nat Commun** 2014;5:4006. (원 Lung1 데이터셋 논문)
- Simon BD, Harmon SA, Turkbey B, et al. *A multimodal automated deep learning-based model for predicting biochemical recurrence of prostate cancer.* **Clin Imaging** 2025;126:110579. (M1~M4 모델 구조 참조)
- Chen RJ, et al. *Pan-cancer integrative histology-genomic analysis via multimodal deep learning.* **Cancer Cell** 2022. (멀티모달 fusion 일반론)
- Selvaraju RR, et al. *Grad-CAM: Visual explanations from deep networks.* **ICCV** 2017.
- Lundberg SM, Lee SI. *A unified approach to interpreting model predictions.* **NeurIPS** 2017. (SHAP)

### 8.3 데이터 인용 (필수)

> Aerts, H. J. W. L., Wee, L., Rios Velazquez, E., Leijenaar, R. T. H., Parmar, C., Grossmann, P., Carvalho, S., Bussink, J., Monshouwer, R., Haibe-Kains, B., Rietveld, D., Hoebers, F., Rietbergen, M. M., Leemans, C. R., Dekker, A., Quackenbush, J., Gillies, R. J., Lambin, P. (2014). **Data From NSCLC-Radiomics (version 4)** [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI

---

## 9. 부록

### 9.1 예상 소프트웨어 스택

```
- Python 3.10
- PyTorch 2.x (CUDA 12)
- MONAI (의료영상 딥러닝 유틸)
- SimpleITK, nibabel (NIfTI I/O)
- PyRadiomics (이미 추출 완료, 재현용으로만 필요)
- scikit-learn, pandas, numpy
- shap, pytorch-grad-cam, captum
- lifelines (KM curve, log-rank)
- wandb
- matplotlib, seaborn
- Streamlit (≥1.30)  # 서비스 데모 UI
```

### 9.2 윤리 및 데이터 사용 규정

- TCIA 데이터는 **de-identified public dataset**이므로 추가 IRB 불필요 (기관 정책에 따라 내부 확인 권장)
- CC BY-NC 3.0 라이선스 — **비상업적 연구 목적에 한함**, 재배포 시 인용 필수
- 발표·문서에서 데이터 citation 누락 금지

### 9.3 성공 기준 요약

| 수준 | 기준 |
|---|---|
| **Minimum** (반드시 달성) | M1~M4 모두 학습 완료, 성능 비교 테이블 제시, Grad-CAM + SHAP 시각화 최소 1건씩, **Streamlit 앱 — 사전 로드 샘플 케이스 1명 이상 시연 가능** |
| **Target** (목표) | M4 AUROC ≥ 0.75, 단일 모달리티 대비 통계적 유의 개선, XAI 정량 평가 1개 이상 (IoU or Pointing game), **앱 — CT 업로드 포함 full flow, 3-tab XAI 모두 동작** |
| **Stretch** (여유 시) | Attention fusion 실험, Faithfulness/Stability 평가, 반복 실험 신뢰구간, **앱 — 실시간 Radiomics 재추출, 3-plane CT viewer 완성도** |

---

*작성일: 2026-04-16*
*문서 버전: v3.0*
*주요 변경:*
- *v2.0: 10일 일정 압축, GPU 단독 사용(RTX 4070 Ti Super) 반영, 데이터 준비 완료 전제, n=419 명시, CT slice shape 512×512 반영*
- *v3.0: **서비스 구현 단계(Streamlit 로컬 데모) 추가** — 3.8절 신설, Day 9 서비스 개발 / Day 10 통합·발표로 재편, 기존 Day 8+9 XAI를 Day 8에 압축, Faithfulness/Stability는 Stretch로 이동*
