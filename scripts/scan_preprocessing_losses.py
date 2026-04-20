"""420명에게 MONAI 전처리 돌려서 mask가 잘려나가는 환자 찾기.

사용: python scripts/scan_preprocessing_losses.py --workers 4
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

# src/ import 가능하도록
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def process_one(image_path: str, mask_path: str, pid: str) -> dict:
    # 각 프로세스마다 transform을 새로 만드는 건 비효율이지만, MONAI transform은
    # 직렬화가 까다로워서 이게 가장 안전.
    from src.data import build_preprocess

    pre = build_preprocess(training=False)
    try:
        out = pre({"image": image_path, "mask": mask_path})
        mask_sum = int(out["mask"].sum())
        return {"patient_id": pid, "mask_sum_after": mask_sum, "status": "ok"}
    except Exception as e:
        return {"patient_id": pid, "mask_sum_after": -1, "status": f"err: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from src.data.dataset import load_manifest

    m = load_manifest()
    print(f"[scan] {len(m)} patients, workers={args.workers}", flush=True)

    lost: list[str] = []
    errs: list[str] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, str(r["image_path"]), str(r["mask_path"]), r["patient_id"]): r["patient_id"]
            for _, r in m.iterrows()
        }
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res["status"] != "ok":
                errs.append(f"{res['patient_id']} ({res['status']})")
            elif res["mask_sum_after"] == 0:
                lost.append(res["patient_id"])
            if done % 40 == 0:
                print(f"  {done}/{len(m)}  lost={len(lost)}  err={len(errs)}", flush=True)

    print(f"\n[RESULT] masks lost after preprocessing: {len(lost)} / {len(m)}", flush=True)
    if lost:
        print(f"affected patients: {sorted(lost)}", flush=True)
    if errs:
        print(f"errors ({len(errs)}): {errs[:10]}", flush=True)


if __name__ == "__main__":
    main()
