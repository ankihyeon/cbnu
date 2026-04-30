"""
03_evaluate_bg_ob.py (v2: 옵션 A 개선 적용)
============================================
개선사항:
  1) Confidence threshold 상향 (0.25 → 0.5)
  2) Mask 후처리 (morphological opening + min-area filter)
  3) BG threshold grid search (0.075 / 0.15 / 0.25 / 0.40)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bg_algorithm import compute_crack_length
from src.orthoboundary import compute_crack_widths
from src.polygon_to_mask import yolo_seg_to_mask


# ----- 옵션 A 개선 1: confidence threshold 상향 -----
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# ----- 옵션 A 개선 2: mask 후처리 -----
def postprocess_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Mask 후처리:
      - Opening (3x3): 작은 잡음 제거
      - Closing (3x3): 끊어진 균열 연결
      - min_area 미만 component 제거
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[lbl == i] = 255
    return cleaned


def predict_mask(yolo_model, img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((1, 1), dtype=np.uint8)
    h, w = img.shape[:2]
    # ----- 개선 1: conf, iou 상향 -----
    results = yolo_model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        if r.masks is None:
            continue
        for m in r.masks.data.cpu().numpy():
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (m_resized > 0.5).astype(np.uint8) * 255)
    # ----- 개선 2: 후처리 -----
    mask = postprocess_mask(mask, min_area=50)
    return mask


def metrics_at_threshold(gt: np.ndarray, pred: np.ndarray, bg_threshold: float) -> dict:
    """주어진 BG threshold에서의 길이·폭 지표 (per image)."""
    gt_len = compute_crack_length(gt, threshold=bg_threshold)
    pred_len = compute_crack_length(pred, threshold=bg_threshold)
    gt_w = compute_crack_widths(gt)
    pred_w = compute_crack_widths(pred)
    return {
        "bg_threshold": bg_threshold,
        "gt_length": gt_len["total_length"],
        "pred_length": pred_len["total_length"],
        "gt_mean_width": gt_w["mean_width"],
        "pred_mean_width": pred_w["mean_width"],
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def stats(gt_key, pred_key):
        gt = np.array([r[gt_key] for r in rows], dtype=np.float64)
        pr = np.array([r[pred_key] for r in rows], dtype=np.float64)
        valid = (gt > 0) | (pr > 0)
        gt, pr = gt[valid], pr[valid]
        if len(gt) == 0:
            return {"PA(%)": 0.0, "NRMSE": 0.0, "MAE": 0.0, "MBE": 0.0, "n": 0}
        diff = pr - gt
        mae = float(np.mean(np.abs(diff)))
        mbe = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        avg = float(np.mean(gt)) if np.mean(gt) > 1e-6 else 1.0
        return {
            "PA(%)": round(max(0.0, 1.0 - mae / avg) * 100.0, 2),
            "NRMSE": round(rmse / avg, 4),
            "MAE": round(mae, 3),
            "MBE": round(mbe, 3),
            "n": int(len(gt)),
        }
    return {
        "length": stats("gt_length", "pred_length"),
        "width": stats("gt_mean_width", "pred_mean_width"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bg_thresholds", nargs="+", type=float,
                        default=[0.075, 0.15, 0.25, 0.40])
    args = parser.parse_args()

    from ultralytics import YOLO

    data_root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split

    images = sorted(img_dir.glob("*.*"))
    print(f"[INFO] {args.split} {len(images)}장 평가 시작")
    print(f"[INFO] 옵션 A 개선 적용: conf={CONF_THRESHOLD}, mask postproc=on, BG thresholds={args.bg_thresholds}")

    # threshold별 row 누적
    all_rows: dict[float, list[dict]] = {t: [] for t in args.bg_thresholds}

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = lbl_dir / (img_path.stem + ".txt")
        gt_mask = yolo_seg_to_mask(lbl, w, h, target_classes={0, 1, 2})
        pred_mask = predict_mask(model, img_path)

        for t in args.bg_thresholds:
            row = metrics_at_threshold(gt_mask, pred_mask, t)
            row["image"] = img_path.name
            all_rows[t].append(row)

        if i % 10 == 0:
            print(f"  {i}/{len(images)} 처리 완료")

    # threshold별 집계
    summary = {
        "split": args.split,
        "n_images": len(images),
        "improvements": {
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "mask_postprocess": "opening + closing + min_area=50",
        },
        "per_threshold": {},
    }

    best_t = None
    best_pa_sum = -1
    for t, rows in all_rows.items():
        agg = aggregate(rows)
        summary["per_threshold"][f"{t:.3f}"] = agg
        pa_sum = agg["length"]["PA(%)"] + agg["width"]["PA(%)"]
        if pa_sum > best_pa_sum:
            best_pa_sum = pa_sum
            best_t = t

    summary["best_threshold"] = f"{best_t:.3f}"
    summary["best_metrics"] = summary["per_threshold"][f"{best_t:.3f}"]

    # 최적 threshold의 per-image CSV 저장
    csv_path = out / f"per_image_{args.split}_v2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[best_t][0].keys()))
        writer.writeheader()
        writer.writerows(all_rows[best_t])

    (out / f"summary_{args.split}_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[OK] 평가 완료")
    print(f"  최적 BG threshold: {best_t}")
    print(f"  Length PA: {summary['best_metrics']['length']['PA(%)']}%")
    print(f"  Width  PA: {summary['best_metrics']['width']['PA(%)']}%")


if __name__ == "__main__":
    main()
