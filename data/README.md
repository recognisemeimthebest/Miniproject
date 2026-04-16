# 데이터 준비 가이드

이 폴더는 `.gitignore`에 의해 **git에 추적되지 않습니다**.  
아래 절차에 따라 각자 로컬에 데이터를 세팅하세요.

---

## 데이터셋: NSCLC-Radiomics (TCIA Lung1)

- **출처:** https://www.cancerimagingarchive.net/collection/nsclc-radiomics/
- **DOI:** 10.7937/K9/TCIA.2015.PF0M9REI
- **라이선스:** CC BY-NC 3.0 (비상업적 연구 목적만 허용, 인용 필수)

---

## 폴더 구조 (로컬 세팅 후 이렇게 맞춰주세요)

```
data/
├── raw/
│   ├── DICOM/              # 원본 DICOM (팀장 워크스테이션에서 공유)
│   └── Lung1.clinical.csv  # 임상 데이터
├── processed/
│   ├── ct_nifti/           # DICOM → NIfTI 변환 완료
│   │   └── LUNG1-001/
│   │       ├── ct.nii.gz
│   │       └── gtv_mask.nii.gz
│   ├── radiomics_features.csv   # PyRadiomics 추출 완료
│   ├── clinical.csv             # 전처리된 임상 변수
│   └── splits_seed42.csv        # Train/Val/Test split 인덱스
```

---

## 전처리 재현 방법 (이미 완료 — 참고용)

```bash
# 1. DICOM → NIfTI 변환
python scripts/dicom_to_nifti.py

# 2. Radiomics 추출
python scripts/extract_radiomics.py

# 3. Train/Val/Test split 생성 (seed=42 고정)
python scripts/create_splits.py
```

---

## 인용 (필수)

> Aerts, H. J. W. L., et al. (2014). *Data From NSCLC-Radiomics (version 4)* [Data set].  
> The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI
