"""
smoke_test.py
=============
큐레이션된 데이터셋의 1번째 이미지에 대해 polygon→mask→BG/OB 파이프라인을
동작시켜 알고리즘이 끝까지 작동하는지 확인.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bg_algorithm import compute_crack_length
from src.orthoboundary import compute_crack_widths
from src.polygon_to_mask import yolo_seg_to_mask
from src.io_utils import imread_unicode, imwrite_unicode


def main():
    root = Path(__file__).resolve().parent.parent
    img_dir = root / "data" / "curated" / "images" / "test"
    lbl_dir = root / "data" / "curated" / "labels" / "test"

    sample = sorted(img_dir.glob("*.*"))[0]
    label = lbl_dir / (sample.stem + ".txt")
    img = imread_unicode(sample)
    h, w = img.shape[:2]

    mask = yolo_seg_to_mask(label, w, h, target_classes={0, 1, 2})
    print(f"이미지: {sample.name}  크기: {w}x{h}  mask 픽셀 비율: {(mask>0).mean():.4f}")

    print("\n[BG 알고리즘]")
    bg = compute_crack_length(mask, threshold=0.075, return_visualization=True)
    print(f"  skeleton 픽셀: {bg['skeleton_pixels']}")
    print(f"  trunk 길이: {bg['trunk_length']:.2f} px")
    print(f"  branch 길이: {bg['branch_length']:.2f} px")
    print(f"  total 길이: {bg['total_length']:.2f} px")
    print(f"  체인 수: {bg['num_chains']} / 통합 branches: {bg['num_branches_used']}")

    out = root / "results" / "smoke_test"
    out.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out / "input_image.jpg", img)
    imwrite_unicode(out / "gt_mask.png", mask)
    if bg["visualization"] is not None:
        imwrite_unicode(out / "bg_visualization.png", bg["visualization"])

    print("\n[OrthoBoundary 알고리즘]")
    ob = compute_crack_widths(mask, sample_interval=10, pca_window=7)
    print(f"  components: {ob['num_components']}")
    print(f"  측정 점수: {ob['num_samples']}")
    print(f"  평균 폭: {ob['mean_width']:.3f} px")
    print(f"  중앙값 폭: {ob['median_width']:.3f} px")
    print(f"  최대 폭: {ob['max_width']:.3f} px")

    print(f"\n[OK] 결과 저장: {out}")


if __name__ == "__main__":
    main()
