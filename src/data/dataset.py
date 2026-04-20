"""NSCLC-Radiomics Lung1 Dataset (CT + GTV mask + 2년 생존 레이블).

NIfTI(`ct.nii.gz`, `gtv_mask.nii.gz`) + 임상 CSV를 조인해서
PyTorch Dataset + train/val/test split을 제공.

기획서 §2.3 레이블 정의:
    Survival.time >= 730 & dead == 0  → Class 1 (2년 이상 생존 확정)
    Survival.time >= 730 & dead == 1  → Class 1 (2년 이후 사망 = 2년 시점엔 생존)
    Survival.time <  730 & dead == 1  → Class 0 (2년 이내 사망 확정)
    Survival.time <  730 & dead == 0  → 제외 (조기 censored, outcome unknown)

사용 예:
    from src.data import build_preprocess
    from src.data.dataset import load_manifest, split_manifest, NSCLCDataset

    manifest = load_manifest()                         # 420명
    splits = split_manifest(manifest, seed=42)         # train/val/test dict
    ds = NSCLCDataset(splits["train"],
                      transform=build_preprocess(training=True))
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

NIFTI_ROOT_DEFAULT = Path("data/processed/ct_nifti")
LABEL_CSV_DEFAULT = Path("metadata/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
TWO_YEARS_DAYS = 730


def load_manifest(
    nifti_root: Path | str = NIFTI_ROOT_DEFAULT,
    label_csv: Path | str = LABEL_CSV_DEFAULT,
) -> pd.DataFrame:
    """임상 CSV + 디스크 NIfTI 교집합으로 manifest 생성.

    - `Survival.time < 730 & dead == 0` (조기 censored) 환자는 제외
    - NIfTI 파일이 디스크에 없는 환자도 제외

    Returns:
        pd.DataFrame with columns:
            patient_id, image_path, mask_path, label (0/1),
            stage, survival_time, dead
    """
    nifti_root = Path(nifti_root)
    df = pd.read_csv(label_csv)

    keep = ~((df["Survival.time"] < TWO_YEARS_DAYS) & (df["deadstatus.event"] == 0))
    df = df.loc[keep].copy()

    df["label"] = (df["Survival.time"] >= TWO_YEARS_DAYS).astype(int)
    df["image_path"] = df["PatientID"].apply(lambda pid: nifti_root / pid / "ct.nii.gz")
    df["mask_path"] = df["PatientID"].apply(lambda pid: nifti_root / pid / "gtv_mask.nii.gz")

    have_both = df["image_path"].apply(Path.exists) & df["mask_path"].apply(Path.exists)
    df = df.loc[have_both]

    out = df.rename(
        columns={
            "PatientID": "patient_id",
            "Survival.time": "survival_time",
            "deadstatus.event": "dead",
            "Overall.Stage": "stage",
        }
    )[["patient_id", "image_path", "mask_path", "label", "stage", "survival_time", "dead"]]
    return out.reset_index(drop=True)


def split_manifest(
    manifest: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify: str = "label",
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split.

    기본 stratify는 label (2 classes) — n=420에서 label×stage (최대 8 strata)는
    일부 cell의 샘플 수가 너무 적어 split이 깨질 수 있어 label only가 안전.

    Args:
        stratify: "label" (기본) 또는 "label_stage" (과잉 분할, 주의).

    Returns:
        {"train": df, "val": df, "test": df}, 각 df는 index reset됨.
    """
    if stratify == "label_stage":
        strata = manifest["label"].astype(str) + "_" + manifest["stage"].astype(str)
    else:
        strata = manifest["label"]

    rest_idx, test_idx = train_test_split(
        manifest.index, test_size=test_frac, stratify=strata, random_state=seed
    )
    rest_strata = strata.loc[rest_idx]
    val_size = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_size, stratify=rest_strata, random_state=seed
    )
    return {
        "train": manifest.loc[train_idx].reset_index(drop=True),
        "val": manifest.loc[val_idx].reset_index(drop=True),
        "test": manifest.loc[test_idx].reset_index(drop=True),
    }


class NSCLCDataset(Dataset):
    """CT + GTV mask + 2년 생존 레이블 PyTorch Dataset.

    __getitem__ 리턴:
        {
            "image": Tensor(1, H, W, D) float32 ∈ [0, 1],
            "mask":  Tensor(1, H, W, D) float32 ∈ {0, 1},
            "label": Tensor() int64 (0 or 1),
            "patient_id": str,
        }

    transform은 MONAI Compose (build_preprocess 리턴값) 전달.
    """

    def __init__(self, manifest: pd.DataFrame, transform: Optional[Callable] = None):
        self.rows = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        sample = {"image": str(row["image_path"]), "mask": str(row["mask_path"])}
        if self.transform is not None:
            sample = self.transform(sample)
        return {
            "image": sample["image"],
            "mask": sample["mask"],
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "patient_id": row["patient_id"],
        }


__all__ = [
    "NSCLCDataset",
    "load_manifest",
    "split_manifest",
    "NIFTI_ROOT_DEFAULT",
    "LABEL_CSV_DEFAULT",
    "TWO_YEARS_DAYS",
]
