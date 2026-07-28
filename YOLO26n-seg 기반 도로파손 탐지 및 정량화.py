"""
02_infer_quantify.py
====================
학습된 YOLO26n-seg 모델로 이미지(또는 주행영상 프레임)를 추론하고
파손 정량화 지표를 이미지 단위로 산출 (프레임워크 모듈 ②~③).

출력:
  - per_image_damage.csv  : 이미지별 파손 지표
  - damage_summary.json   : 전체 집계

※ GPS 구간 매핑·구간 단위 Feature 집계·PCI 산출은 설계 단계이며
  `docs/FRAMEWORK_DESIGN.md` 참조. 본 스크립트는 그 입력이 되는
  이미지 단위 정량화까지를 담당한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent / "캡스톤_프로젝트2"
sys.path.insert(0, str(PROJECT_ROOT))

from src.damage_quantify import quantify_image
from src.polygon_to_mask import DAMAGE_CLASSES, CLASS_NAMES

CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5


def yolo_result_to_class_masks(result, h: int, w: int) -> dict[int, np.ndarray]:
    """Ultralytics 결과 1건 → {class_id: mask(uint8 0/255)} (파손 클래스만)."""
    masks: dict[int, np.ndarray] = {}
    if result.masks is None:
        return masks
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    mdata = result.masks.data.cpu().numpy()
    for cls_id, m in zip(cls_ids, mdata):
        if cls_id not in DAMAGE_CLASSES:
            continue
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        bin_m = (m_resized > 0.5).astype(np.uint8) * 255
        if cls_id in masks:
            masks[cls_id] = np.maximum(masks[cls_id], bin_m)
        else:
            masks[cls_id] = bin_m
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="best.pt 경로")
    parser.add_argument("--images", required=True, help="추론할 이미지 폴더")
    parser.add_argument("--output", default="results", help="결과 출력 폴더")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--min_area", type=int, default=30)
    args = parser.parse_args()

    from ultralytics import YOLO

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.images)
    images = sorted([p for p in img_dir.glob("*.*")
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise SystemExit(f"이미지 없음: {img_dir}")

    model = YOLO(args.weights)
    print(f"[INFO] {len(images)}장 추론 + 정량화 (conf={args.conf}, iou={args.iou})")

    rows = []
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        result = model(img, conf=args.conf, iou=args.iou, verbose=False)[0]
        class_masks = yolo_result_to_class_masks(result, h, w)
        q = quantify_image(class_masks, (h, w), min_area=args.min_area)

        row = {
            "image": img_path.name,
            "damage_area_ratio": round(q["total_damage_area_ratio"], 6),
            "damage_count": q["total_damage_count"],
            "crack_length_px": round(q["total_crack_length_px"], 2),
            "severity_score": round(q["severity_weighted_score"], 6),
        }
        # 클래스별 면적비
        for cls_id in sorted(DAMAGE_CLASSES):
            name = CLASS_NAMES[cls_id]
            pc = q["per_class"].get(name)
            row[f"{name}_area_ratio"] = round(pc["area_ratio"], 6) if pc else 0.0
        rows.append(row)

        if i % 20 == 0:
            print(f"  {i}/{len(images)}")

    csv_path = out / "per_image_damage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    arr_area = np.array([r["damage_area_ratio"] for r in rows])
    arr_sev = np.array([r["severity_score"] for r in rows])
    summary = {
        "n_images": len(rows),
        "weights": str(args.weights),
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "mean_damage_area_ratio": round(float(arr_area.mean()), 6),
        "mean_severity_score": round(float(arr_sev.mean()), 6),
        "images_with_damage": int((arr_area > 0).sum()),
    }
    (out / "damage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] 정량화 완료 → {csv_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
