"""
polygon_to_mask.py
==================
YOLO segmentation polygon (정규화 좌표) → 클래스별 binary mask 변환.

YOLO seg 라벨 포맷:
    class_id x1 y1 x2 y2 ... xn yn   (모든 좌표는 [0, 1] 정규화)

본 프로젝트(#2)는 8개 클래스(균열 3종 + human/manhole/pothole/sewer/vehicle)를
다루므로, 클래스별로 분리된 마스크를 반환할 수 있도록 한다.
파손 정량화(`damage_quantify.py`)는 이 중 파손 클래스(균열·포트홀)만 사용한다.
"""

from pathlib import Path

import numpy as np
import cv2


# data.yaml 기준 클래스 인덱스
CLASS_NAMES = [
    "Crack_Crazing", "Crack_Longitudinal", "Crack_Transverse",
    "human", "manhole", "pothole", "sewer", "vehicle",
]
# 도로 파손으로 간주하는 클래스 (정량화·상태평가 대상)
CRACK_CLASSES = {0, 1, 2}
POTHOLE_CLASSES = {5}
DAMAGE_CLASSES = CRACK_CLASSES | POTHOLE_CLASSES


def yolo_seg_to_masks(label_path: Path, img_w: int, img_h: int,
                      target_classes: set[int] | None = None
                      ) -> dict[int, np.ndarray]:
    """
    YOLO segmentation label → {class_id: binary mask(uint8 0/255)} 딕셔너리.

    Args:
        label_path: YOLO seg .txt 파일
        img_w, img_h: 이미지 크기
        target_classes: 포함할 class id 집합 (None=전체)

    Returns:
        masks: {class_id: (H, W) uint8 0/255}. 등장하지 않은 클래스는 키 없음.
    """
    masks: dict[int, np.ndarray] = {}
    if not label_path.exists():
        return masks

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # class + 최소 3점(x,y) = 7개 필드
            continue
        cls_id = int(parts[0])
        if target_classes is not None and cls_id not in target_classes:
            continue

        coords = list(map(float, parts[1:]))
        polygon = np.array([
            [coords[i] * img_w, coords[i + 1] * img_h]
            for i in range(0, len(coords) - 1, 2)
        ], dtype=np.int32)
        if len(polygon) < 3:
            continue

        if cls_id not in masks:
            masks[cls_id] = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(masks[cls_id], [polygon], 255)

    return masks


def yolo_seg_to_mask(label_path: Path, img_w: int, img_h: int,
                     target_classes: set[int] | None = None) -> np.ndarray:
    """target_classes를 하나로 합친 단일 binary mask (0/255)."""
    masks = yolo_seg_to_masks(label_path, img_w, img_h, target_classes)
    out = np.zeros((img_h, img_w), dtype=np.uint8)
    for m in masks.values():
        out = np.maximum(out, m)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()

    masks = yolo_seg_to_masks(Path(args.label), args.width, args.height)
    for cls_id, m in sorted(masks.items()):
        ratio = (m > 0).mean()
        print(f"  class {cls_id} ({CLASS_NAMES[cls_id]}): area_ratio={ratio:.5f}")
