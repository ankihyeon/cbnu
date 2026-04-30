"""
polygon_to_mask.py
==================
YOLO segmentation polygon (정규화된 좌표) → binary mask 변환.

YOLO seg 라벨 포맷:
    class_id x1 y1 x2 y2 ... xn yn   (모든 좌표는 [0, 1] 정규화)

Li et al. 2024의 BG/OrthoBoundary 알고리즘은 binary mask가 필요하므로
이 변환이 첫 단계.
"""

from pathlib import Path
import numpy as np
import cv2


def yolo_seg_to_mask(label_path: Path, img_w: int, img_h: int,
                     target_classes: set[int] | None = None) -> np.ndarray:
    """
    YOLO segmentation label → binary mask (uint8, 0/255).

    Args:
        label_path: YOLO seg .txt 파일
        img_w, img_h: 이미지 크기
        target_classes: 마스크에 포함할 class id 집합 (None=전체)

    Returns:
        mask: shape (H, W), dtype uint8, 0(배경) / 255(균열)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # class + 최소 3점(x,y) = 7개 필드
            continue
        cls_id = int(parts[0])
        if target_classes is not None and cls_id not in target_classes:
            continue

        coords = list(map(float, parts[1:]))
        # [x1, y1, x2, y2, ...] → [(x1*W, y1*H), ...]
        polygon = np.array([
            [coords[i] * img_w, coords[i + 1] * img_h]
            for i in range(0, len(coords) - 1, 2)
        ], dtype=np.int32)

        if len(polygon) < 3:
            continue
        cv2.fillPoly(mask, [polygon], 255)

    return mask


def load_image_and_mask(img_path: Path, label_path: Path,
                        target_classes: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """이미지와 mask를 동시 로드."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {img_path}")
    h, w = img.shape[:2]
    mask = yolo_seg_to_mask(label_path, w, h, target_classes)
    return img, mask


def batch_convert(label_dir: Path, image_dir: Path, mask_dir: Path,
                  target_classes: set[int] | None = None) -> int:
    """폴더 단위 일괄 변환."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for label in sorted(label_dir.glob("*.txt")):
        # 이미지 확장자 자동 매칭
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = image_dir / (label.stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = yolo_seg_to_mask(label, w, h, target_classes)
        out = mask_dir / (label.stem + ".png")
        cv2.imwrite(str(out), mask)
        count += 1
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--classes", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    n = batch_convert(
        Path(args.labels), Path(args.images), Path(args.masks),
        target_classes=set(args.classes)
    )
    print(f"✅ {n}개 mask 생성 → {args.masks}")
