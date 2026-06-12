"""
smoke_test.py
=============
학습 없이 GT polygon 라벨로 polygon→mask→파손 정량화 파이프라인을
동작시켜 알고리즘이 끝까지 작동하는지 확인.

YOLO 가중치가 필요 없으므로 데이터셋만 있으면 즉시 실행 가능.
파손 클래스(균열·포트홀)가 포함된 첫 라벨을 자동 탐색한다.

사용법:
    python scripts/smoke_test.py --data_root D:/YOLO_data_v3 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_utils import imread_unicode, imwrite_unicode
from src.polygon_to_mask import yolo_seg_to_masks, DAMAGE_CLASSES, CLASS_NAMES
from src.damage_quantify import quantify_image


def find_damage_sample(img_dir: Path, lbl_dir: Path) -> tuple[Path, Path] | None:
    """파손 클래스가 포함된 첫 (이미지, 라벨) 쌍 탐색."""
    for img in sorted(img_dir.glob("*.*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and int(parts[0]) in DAMAGE_CLASSES:
                return img, lbl
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="D:/YOLO_data_v3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/smoke_test")
    args = parser.parse_args()

    root = Path(args.data_root)
    img_dir = root / args.split / "images"
    lbl_dir = root / args.split / "labels"
    if not img_dir.exists():
        raise SystemExit(f"이미지 폴더 없음: {img_dir}")

    found = find_damage_sample(img_dir, lbl_dir)
    if found is None:
        raise SystemExit(f"{args.split}에 파손 클래스 라벨이 있는 이미지를 찾지 못함")
    img_path, lbl_path = found

    img = imread_unicode(img_path)
    h, w = img.shape[:2]
    masks = yolo_seg_to_masks(lbl_path, w, h, target_classes=DAMAGE_CLASSES)

    print(f"이미지: {img_path.name}  크기: {w}x{h}")
    print(f"검출된 파손 클래스: {[CLASS_NAMES[c] for c in sorted(masks)]}")

    result = quantify_image(masks, (h, w))
    print("\n[파손 정량화]")
    print(f"  총 파손 면적비: {result['total_damage_area_ratio']:.5f}")
    print(f"  총 파손 개수:   {result['total_damage_count']}")
    print(f"  총 균열 길이:   {result['total_crack_length_px']:.1f} px")
    print(f"  심각도 가중점수: {result['severity_weighted_score']:.5f}")
    for name, info in result["per_class"].items():
        print(f"   - {name}: count={info['instance_count']} "
              f"area_ratio={info['area_ratio']:.5f} "
              f"len={info['length_px']:.1f}px width={info['mean_width_px']:.2f}px")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out / "input_image.jpg", img)
    # 파손 클래스 통합 마스크 시각화
    import numpy as np
    merged = np.zeros((h, w), dtype=np.uint8)
    for m in masks.values():
        merged = np.maximum(merged, m)
    imwrite_unicode(out / "gt_damage_mask.png", merged)
    (out / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] 결과 저장: {out}")


if __name__ == "__main__":
    main()
