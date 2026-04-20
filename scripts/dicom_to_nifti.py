"""DICOM → NIfTI 변환 스크립트 (NSCLC-Radiomics 전용)

입력:  data/raw/ 또는 metadata/nsclc_radiomics/ 아래 환자별 DICOM 폴더
출력:  data/processed/ct_nifti/LUNG1-xxx/
         ├── ct.nii.gz          # 원본 HU 보존된 3D CT (int16)
         └── gtv_mask.nii.gz    # CT와 동일 shape·affine인 binary mask (uint8)

CT와 mask의 좌표계(affine)를 일치시켜 저장하므로 C의 PyRadiomics
파이프라인에서 바로 사용 가능.

사용법:
    python scripts/dicom_to_nifti.py --src metadata/nsclc_radiomics --dst data/processed/ct_nifti
    python scripts/dicom_to_nifti.py --src metadata/nsclc_radiomics --dst data/processed/ct_nifti --patients LUNG1-001 LUNG1-002
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def scan_series(patient_dir: Path) -> dict[str, list[Path]]:
    """환자 폴더 아래의 series를 Modality 기준으로 분류.

    returns:
        {"CT": [dcm path, ...], "RTSTRUCT": [...], "SEG": [...]}
    """
    result: dict[str, list[Path]] = {"CT": [], "RTSTRUCT": [], "SEG": []}
    for series_dir in _iter_series(patient_dir):
        dcm_files = sorted(series_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        except Exception as e:
            log.warning(f"  read fail: {series_dir.name}: {e}")
            continue
        modality = getattr(ds, "Modality", "")
        if modality == "CT":
            result["CT"] = dcm_files
        elif modality == "RTSTRUCT":
            result["RTSTRUCT"] = dcm_files
        elif modality == "SEG":
            result["SEG"] = dcm_files
    return result


def _iter_series(patient_dir: Path):
    for study in patient_dir.iterdir():
        if not study.is_dir():
            continue
        for series in study.iterdir():
            if series.is_dir():
                yield series


def load_ct(ct_dir: Path) -> sitk.Image:
    """DICOM CT series → SimpleITK 3D 볼륨 (HU 원값 보존)."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {ct_dir}")
    files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
    reader.SetFileNames(files)
    img = reader.Execute()
    if img.GetPixelID() != sitk.sitkInt16:
        img = sitk.Cast(img, sitk.sitkInt16)
    return img


def load_mask_seg(seg_path: Path, reference: sitk.Image) -> sitk.Image:
    """DICOM SEG → binary mask (reference CT와 같은 shape·affine으로 정렬).

    GTV-1이 여러 segment로 나뉘어 있으면 전부 OR로 합침.
    """
    seg_img = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(seg_img)
    if arr.ndim == 4:
        mask_arr = (arr.sum(axis=0) > 0).astype(np.uint8)
    else:
        mask_arr = (arr > 0).astype(np.uint8)

    mask = sitk.GetImageFromArray(mask_arr)
    mask.SetSpacing(seg_img.GetSpacing())
    mask.SetDirection(seg_img.GetDirection())
    mask.SetOrigin(seg_img.GetOrigin())

    return _resample_to_reference(mask, reference, is_label=True)


def load_mask_rtstruct(rt_path: Path, reference: sitk.Image, ct_dir: Path) -> sitk.Image:
    """DICOM RTSTRUCT → binary mask (reference CT와 같은 shape·affine).

    rt-utils 사용. GTV-1 contour 선호, 없으면 GTV로 시작하는 첫 ROI.
    """
    from rt_utils import RTStructBuilder

    rt = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rt_path))
    roi_names = rt.get_roi_names()
    target = _pick_gtv_roi(roi_names)
    if target is None:
        raise RuntimeError(f"No GTV-like ROI in {rt_path.name}. ROIs={roi_names}")

    mask_arr = rt.get_roi_mask_by_name(target).astype(np.uint8)
    mask_arr = np.transpose(mask_arr, (2, 0, 1))

    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(reference)
    return mask


def _pick_gtv_roi(names: list[str]) -> Optional[str]:
    for target in ("GTV-1", "GTV_1", "GTV1"):
        if target in names:
            return target
    for n in names:
        if n.upper().startswith("GTV"):
            return n
    return None


def _resample_to_reference(mask: sitk.Image, reference: sitk.Image, is_label: bool) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)


def convert_patient(patient_dir: Path, dst_dir: Path) -> dict:
    """한 환자 DICOM → NIfTI 변환. 성공/실패 메타를 dict로 반환."""
    pid = patient_dir.name
    info = {"patient": pid, "status": "fail", "ct_shape": None, "mask_sum": None, "reason": ""}

    series = scan_series(patient_dir)
    if not series["CT"]:
        info["reason"] = "no CT series"
        return info

    try:
        ct = load_ct(series["CT"][0].parent)
    except Exception as e:
        info["reason"] = f"CT load: {e}"
        return info

    mask: Optional[sitk.Image] = None
    if series["SEG"]:
        try:
            mask = load_mask_seg(series["SEG"][0], ct)
        except Exception as e:
            log.warning(f"[{pid}] SEG load fail ({e}); fallback to RTSTRUCT")

    if mask is None and series["RTSTRUCT"]:
        try:
            mask = load_mask_rtstruct(series["RTSTRUCT"][0], ct, series["CT"][0].parent)
        except Exception as e:
            info["reason"] = f"RTSTRUCT load: {e}"
            return info

    if mask is None:
        info["reason"] = "no SEG/RTSTRUCT"
        return info

    out_patient = dst_dir / pid
    out_patient.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct, str(out_patient / "ct.nii.gz"), useCompression=True)
    sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(out_patient / "gtv_mask.nii.gz"), useCompression=True)

    ct_arr = sitk.GetArrayFromImage(ct)
    mask_arr = sitk.GetArrayFromImage(mask)
    info.update(
        status="ok",
        ct_shape=list(ct_arr.shape),
        ct_hu_min=int(ct_arr.min()),
        ct_hu_max=int(ct_arr.max()),
        mask_sum=int(mask_arr.sum()),
        spacing=list(ct.GetSpacing()),
    )
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="DICOM 루트 (환자 폴더들이 바로 아래)")
    ap.add_argument("--dst", type=Path, required=True, help="NIfTI 출력 루트")
    ap.add_argument("--patients", nargs="*", default=None, help="특정 환자만 변환 (생략 시 전부)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N명만 처리 (스모크 테스트용)")
    ap.add_argument("--start", type=int, default=None, help="정렬된 환자 리스트의 시작 index (포함)")
    ap.add_argument("--end", type=int, default=None, help="정렬된 환자 리스트의 끝 index (미포함)")
    ap.add_argument("--skip-existing", action="store_true", help="이미 ct.nii.gz + gtv_mask.nii.gz 있으면 건너뜀 (재시도/병렬 안전)")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    all_patients = sorted(p for p in args.src.iterdir() if p.is_dir())
    if args.patients:
        wanted = set(args.patients)
        all_patients = [p for p in all_patients if p.name in wanted]
    if args.start is not None or args.end is not None:
        all_patients = all_patients[args.start or 0 : args.end]
    if args.limit:
        all_patients = all_patients[: args.limit]

    if args.skip_existing:
        before = len(all_patients)
        all_patients = [
            p for p in all_patients
            if not ((args.dst / p.name / "ct.nii.gz").exists() and (args.dst / p.name / "gtv_mask.nii.gz").exists())
        ]
        log.info(f"Skip-existing: {before - len(all_patients)} already done, {len(all_patients)} remaining")

    log.info(f"Total patients to process: {len(all_patients)}")

    results = []
    for idx, p in enumerate(all_patients, 1):
        log.info(f"[{idx}/{len(all_patients)}] {p.name}")
        try:
            info = convert_patient(p, args.dst)
        except Exception as e:
            info = {"patient": p.name, "status": "fail", "reason": f"uncaught: {e}"}
            log.error(traceback.format_exc())
        results.append(info)
        if info["status"] == "ok":
            log.info(f"  OK shape={info['ct_shape']} HU=[{info['ct_hu_min']},{info['ct_hu_max']}] mask_voxels={info['mask_sum']}")
        else:
            log.warning(f"  FAIL: {info['reason']}")

    import csv, socket
    host = socket.gethostname().replace(" ", "_")
    log_path = args.dst / f"conversion_log_{host}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["patient", "status", "ct_shape", "ct_hu_min", "ct_hu_max", "mask_sum", "spacing", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    ok = sum(1 for r in results if r["status"] == "ok")
    log.info(f"Done. {ok}/{len(results)} succeeded. Log: {log_path}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
