"""
damage_quantify.py
==================
파손 정량화 모듈 (프레임워크 모듈 ②~③ 중 "파손 정량화").

YOLO26-seg가 산출한 클래스별 mask(또는 GT polygon mask)로부터
이미지 단위 정량 지표를 산출한다. 이 지표는 이후 (설계 단계인)
GPS 구간 매핑 → 구간 단위 Feature 구조화의 입력이 된다.

산출 지표 (발표자료 슬라이드 7 "파손 정량화" 행 기준):
  - 파손 면적(픽셀) / 균열 점유율(area ratio)
  - 파손 개수(인스턴스/컴포넌트 수)
  - 균열 길이·폭 (skeletonization 기반)
  - 클래스별 + 전체 집계
  - 심각도 가중 손상 점수(severity-weighted damage score)

균열 길이·폭의 정밀 산출은 프로젝트 #1의 BG / Optimized OrthoBoundary
알고리즘(Li et al. 2024)을 채택하며, 본 모듈은 그 경량 근사
(medial-axis 기반)를 제공한다. 정밀 측정이 필요하면 #1 모듈을 연결한다.
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.morphology import medial_axis

from .polygon_to_mask import (
    CLASS_NAMES, CRACK_CLASSES, POTHOLE_CLASSES, DAMAGE_CLASSES,
)


# 클래스별 심각도 가중치 (손상 점수 산출용, 발표자료 슬라이드 8 "심각도").
# 포트홀은 균열보다 통행 안전 위험이 크므로 더 높은 가중치를 부여.
DEFAULT_SEVERITY = {
    0: 1.0,   # Crack_Crazing (망상)
    1: 0.8,   # Crack_Longitudinal (종)
    2: 0.8,   # Crack_Transverse (횡)
    5: 2.5,   # pothole
}


def _skeleton_length(mask_bool: np.ndarray) -> tuple[float, float]:
    """
    medial-axis 기반 균열 길이·평균 폭 근사.

    - 길이: skeleton 픽셀을 8-이웃 Euclidean(직선 1.0 / 대각 √2)로 누적
    - 폭: skeleton 위 거리변환 값의 평균 × 2

    Returns:
        (length_px, mean_width_px)
    """
    if mask_bool.sum() == 0:
        return 0.0, 0.0
    skel, dist = medial_axis(mask_bool, return_distance=True)
    n_skel = int(skel.sum())
    if n_skel == 0:
        return 0.0, 0.0

    # 길이: 대각 연결 비율을 고려한 보정 (대각 이웃 1개당 √2 - 1 가산)
    # 단순·견고한 근사: 픽셀 수 + 대각 보정
    diag_kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.uint8)
    skel_u8 = skel.astype(np.uint8)
    diag_neighbors = cv2.filter2D(skel_u8, -1, diag_kernel,
                                  borderType=cv2.BORDER_CONSTANT)
    diag_count = float((diag_neighbors[skel] / 2.0).sum())  # 양방향 중복 제거
    length_px = float(n_skel) + diag_count * (np.sqrt(2.0) - 1.0)

    mean_width = float(dist[skel].mean()) * 2.0
    return length_px, mean_width


def quantify_class_mask(mask: np.ndarray, cls_id: int,
                        min_area: int = 30) -> dict:
    """
    단일 클래스 mask의 정량 지표.

    Args:
        mask: (H, W) uint8 0/255 또는 bool
        cls_id: 클래스 인덱스
        min_area: 이보다 작은 컴포넌트는 잡음으로 제거

    Returns:
        지표 dict
    """
    bm = (mask > 0).astype(np.uint8)
    h, w = bm.shape
    img_area = float(h * w)

    num, lbl, stats, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
    kept = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    total_area = float(sum(stats[i, cv2.CC_STAT_AREA] for i in kept))

    info = {
        "class_id": cls_id,
        "class_name": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id),
        "instance_count": len(kept),
        "area_px": total_area,
        "area_ratio": total_area / img_area if img_area > 0 else 0.0,
        "length_px": 0.0,
        "mean_width_px": 0.0,
    }

    # 균열 클래스만 길이·폭 산출 (포트홀은 면형이라 면적·등가지름이 의미 있음)
    if cls_id in CRACK_CLASSES and kept:
        clean = np.zeros_like(bm)
        for i in kept:
            clean[lbl == i] = 1
        length_px, mean_width = _skeleton_length(clean.astype(bool))
        info["length_px"] = length_px
        info["mean_width_px"] = mean_width
    elif cls_id in POTHOLE_CLASSES and kept:
        # 포트홀 등가지름(circle-equivalent diameter)
        info["equiv_diameter_px"] = float(
            np.mean([2.0 * np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi) for i in kept])
        )

    return info


def quantify_image(class_masks: dict[int, np.ndarray],
                   img_shape: tuple[int, int],
                   severity: dict[int, float] | None = None,
                   min_area: int = 30) -> dict:
    """
    이미지 단위 파손 정량화 (파손 클래스만 집계).

    Args:
        class_masks: {class_id: mask}. polygon_to_mask.yolo_seg_to_masks() 또는
                     YOLO 추론 결과에서 생성.
        img_shape: (H, W)
        severity: 클래스별 심각도 가중치 (None=DEFAULT_SEVERITY)
        min_area: 컴포넌트 최소 면적

    Returns:
        {
          "per_class": {class_name: {...}},
          "total_damage_area_ratio": float,
          "total_damage_count": int,
          "total_crack_length_px": float,
          "severity_weighted_score": float,   # Σ severity * area_ratio
        }
    """
    severity = severity or DEFAULT_SEVERITY
    h, w = img_shape

    per_class = {}
    total_area_ratio = 0.0
    total_count = 0
    total_crack_len = 0.0
    sev_score = 0.0

    for cls_id, mask in class_masks.items():
        if cls_id not in DAMAGE_CLASSES:
            continue
        m = mask if mask.shape == (h, w) else cv2.resize(
            mask, (w, h), interpolation=cv2.INTER_NEAREST)
        info = quantify_class_mask(m, cls_id, min_area=min_area)
        if info["instance_count"] == 0:
            continue
        per_class[info["class_name"]] = info
        total_area_ratio += info["area_ratio"]
        total_count += info["instance_count"]
        total_crack_len += info["length_px"]
        sev_score += severity.get(cls_id, 1.0) * info["area_ratio"]

    return {
        "image_h": h,
        "image_w": w,
        "per_class": per_class,
        "total_damage_area_ratio": total_area_ratio,
        "total_damage_count": total_count,
        "total_crack_length_px": total_crack_len,
        "severity_weighted_score": sev_score,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from .polygon_to_mask import yolo_seg_to_masks
    from .io_utils import imread_unicode

    parser = argparse.ArgumentParser(description="GT polygon 라벨 기반 파손 정량화 데모")
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    img = imread_unicode(args.image)
    if img is None:
        raise SystemExit(f"이미지 로드 실패: {args.image}")
    h, w = img.shape[:2]
    masks = yolo_seg_to_masks(Path(args.label), w, h, target_classes=DAMAGE_CLASSES)
    result = quantify_image(masks, (h, w))
    print(json.dumps(result, ensure_ascii=False, indent=2))
