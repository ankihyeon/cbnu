"""
bg_algorithm.py
===============
Branch Growing (BG) 알고리즘 구현.

논문: Li et al. (2024) "Automated quantification of crack length and width
       in asphalt pavements", Computer-Aided Civil and Infrastructure
       Engineering, 39(22), 3317-3336. DOI: 10.1111/mice.13344

Section 3 의사코드 기반 직접 구현 (저자 코드 미공개).

핵심 단계 (Figure 2):
  1) MAT skeleton 추출 (single-pixel-width)
  2) Connection matrix 구성 (8-neighborhood)
  3) Special structure / self-connection 처리
  4) Graph converting (chains → weighted multigraph → DAG)
  5) Unbiased detector (Steger 1998) 기반 길이 가중치 산출
  6) Skeleton growing: trunk + threshold(0.075) 이상 branches 통합
  7) 최종 길이 = trunk + 통합된 branches

용법:
    from src.bg_algorithm import compute_crack_length
    length, viz = compute_crack_length(binary_mask, threshold=0.075)
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.morphology import medial_axis


# ----------------------------------------------------------------------
# 1) Skeleton 추출 (논문 §3.1.1)
# ----------------------------------------------------------------------
def extract_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    MAT(Medial Axis Transform) 기반 single-pixel-width skeleton.

    Args:
        binary_mask: (H, W) uint8, 0/255 또는 0/1
    Returns:
        skeleton: (H, W) bool
    """
    bm = (binary_mask > 0).astype(np.uint8)
    skel = medial_axis(bm)  # bool array
    return skel


# ----------------------------------------------------------------------
# 2) Connection matrix (논문 §3.1.2)
# ----------------------------------------------------------------------
def get_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    """skeleton 픽셀 좌표 (N, 2) [row, col]."""
    return np.argwhere(skeleton)


def neighbor8_offsets() -> np.ndarray:
    """8-neighborhood relative offsets."""
    return np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],          [0, 1],
        [1, -1],  [1, 0], [1, 1],
    ], dtype=np.int32)


def fix_special_2x2(skeleton: np.ndarray) -> np.ndarray:
    """
    논문 Figure 4 처리: 2×2 블록에서 양 대각선이 동시에 존재하는 경우
    한 대각선의 연결을 제거하여 disordered structure 방지.
    """
    skel = skeleton.copy()
    h, w = skel.shape
    # padding으로 경계 처리
    pad = np.pad(skel, 1, mode="constant", constant_values=False)
    for i in range(h - 1):
        for j in range(w - 1):
            # 2x2 블록
            block = pad[i + 1:i + 3, j + 1:j + 3]
            # 모든 4픽셀 ON & 대각선 패턴
            if block.all():
                # 한쪽 대각 disconnect (관습적으로 우상-좌하 제거)
                skel[i, j + 1] = False
    return skel


def degree_map(skeleton: np.ndarray) -> np.ndarray:
    """
    각 skeleton 픽셀의 degree(이웃 ON 픽셀 수, 자기 자신 제외).
    Returns:
        deg: (H, W) int, skeleton OFF는 0
    """
    skel_f = skeleton.astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    # convolution: 3x3 합 - 자기 자신
    summed = cv2.filter2D(skel_f, ddepth=cv2.CV_32F, kernel=kernel,
                          borderType=cv2.BORDER_CONSTANT)
    deg = (summed - skel_f).astype(np.int32)  # 자기 자신 제외
    deg[~skeleton] = 0
    return deg


# ----------------------------------------------------------------------
# 3) Endpoint / Junction 분류 (논문 §3.1.3)
# ----------------------------------------------------------------------
def classify_points(deg: np.ndarray, skeleton: np.ndarray):
    """
    Returns:
        endpoints: deg == 1 좌표
        relays:    deg == 2 좌표
        junctions: deg >= 3 좌표
    """
    endpoints = np.argwhere(skeleton & (deg == 1))
    relays = np.argwhere(skeleton & (deg == 2))
    junctions = np.argwhere(skeleton & (deg >= 3))
    return endpoints, relays, junctions


# ----------------------------------------------------------------------
# 4) Chain 추출: endpoint-junction 또는 junction-junction 사이 경로
# ----------------------------------------------------------------------
def trace_chains(skeleton: np.ndarray, deg: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Skeleton을 chain(체인) 단위로 분해.
    각 chain은 endpoint/junction에서 시작-끝, 사이는 모두 relay(deg==2).
    """
    visited_edges = set()  # 무방향 엣지: frozenset({(r1,c1), (r2,c2)})
    chains: list[list[tuple[int, int]]] = []
    offs = neighbor8_offsets()

    skel_set = {tuple(p) for p in get_skeleton_points(skeleton)}

    def neighbors(p: tuple[int, int]):
        r, c = p
        out = []
        for dr, dc in offs:
            np_ = (r + dr, c + dc)
            if np_ in skel_set:
                out.append(np_)
        return out

    # 모든 endpoint + junction에서 출발해 chain 추출
    starts = []
    for p in skel_set:
        if deg[p] == 1 or deg[p] >= 3:
            starts.append(p)

    for start in starts:
        for nb in neighbors(start):
            edge = frozenset({start, nb})
            if edge in visited_edges:
                continue
            chain = [start]
            prev = start
            curr = nb
            while True:
                chain.append(curr)
                visited_edges.add(frozenset({prev, curr}))
                if deg[curr] != 2:
                    # endpoint 또는 junction → chain 종료
                    break
                # relay → 다음 픽셀 (이전 제외)
                next_candidates = [n for n in neighbors(curr) if n != prev]
                if not next_candidates:
                    break
                prev = curr
                curr = next_candidates[0]
            chains.append(chain)
    return chains


# ----------------------------------------------------------------------
# 5) Chain 길이 (Steger unbiased detector 단순화 버전)
# ----------------------------------------------------------------------
def chain_length(chain: list[tuple[int, int]]) -> float:
    """
    인접 픽셀 간 Euclidean distance 누적합 (논문 Eq. 2~4 단순화).
    실제 unbiased detector는 더 복잡한 fitting을 수행하지만,
    pixel-width=1 skeleton에서는 누적 Euclidean이 좋은 근사.
    """
    if len(chain) < 2:
        return 0.0
    arr = np.array(chain, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))


# ----------------------------------------------------------------------
# 6) Trunk 추출 + Branch threshold 적용 (논문 §3.1.6)
# ----------------------------------------------------------------------
def find_trunk_and_branches(chains: list[list[tuple[int, int]]],
                            threshold: float = 0.075) -> tuple[float, float, list[int]]:
    """
    Trunk: 가장 긴 단일 chain (단순화: 연결된 chain 시퀀스로 확장 가능)
    Branch: trunk 외 chain 중 길이 > threshold * trunk_length 인 것만 통합.

    Returns:
        trunk_length, total_length, branch_indices_used
    """
    if not chains:
        return 0.0, 0.0, []

    lengths = [chain_length(c) for c in chains]
    trunk_idx = int(np.argmax(lengths))
    trunk_len = lengths[trunk_idx]

    branches_used = []
    branch_total = 0.0
    for i, L in enumerate(lengths):
        if i == trunk_idx:
            continue
        if L > threshold * trunk_len:
            branches_used.append(i)
            branch_total += L

    total = trunk_len + branch_total
    return trunk_len, total, branches_used


# ----------------------------------------------------------------------
# 7) 최상위 API
# ----------------------------------------------------------------------
def compute_crack_length(binary_mask: np.ndarray,
                         threshold: float = 0.075,
                         return_visualization: bool = False
                         ) -> dict:
    """
    Binary mask에서 BG 알고리즘으로 균열 길이 산출.

    Args:
        binary_mask: (H, W) uint8 or bool
        threshold: branch 통합 임계값 (논문 권장값 0.075)
        return_visualization: True 시 시각화 이미지(BGR) 반환

    Returns:
        {
            "skeleton_pixels": int,
            "trunk_length": float,
            "branch_length": float,
            "total_length": float,
            "num_chains": int,
            "num_branches_used": int,
            "visualization": np.ndarray | None,
        }
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask > 0

    skeleton = extract_skeleton(binary_mask.astype(np.uint8) * 255)
    skeleton = fix_special_2x2(skeleton)
    deg = degree_map(skeleton)
    chains = trace_chains(skeleton, deg)
    trunk_len, total_len, branches_used = find_trunk_and_branches(chains, threshold)

    result = {
        "skeleton_pixels": int(skeleton.sum()),
        "trunk_length": trunk_len,
        "branch_length": total_len - trunk_len,
        "total_length": total_len,
        "num_chains": len(chains),
        "num_branches_used": len(branches_used),
        "visualization": None,
    }

    if return_visualization:
        viz = cv2.cvtColor(binary_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        viz[skeleton] = (0, 0, 255)  # 전체 skeleton: 빨강
        if chains:
            lengths = [chain_length(c) for c in chains]
            trunk_idx = int(np.argmax(lengths))
            for i, chain in enumerate(chains):
                color = (0, 255, 0) if i == trunk_idx else (
                    (255, 200, 0) if i in branches_used else (128, 128, 128)
                )
                for r, c in chain:
                    if 0 <= r < viz.shape[0] and 0 <= c < viz.shape[1]:
                        viz[r, c] = color
        result["visualization"] = viz

    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, help="binary mask 이미지 경로")
    parser.add_argument("--threshold", type=float, default=0.075)
    parser.add_argument("--viz_out", default=None)
    args = parser.parse_args()

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"마스크 로드 실패: {args.mask}")

    result = compute_crack_length(mask, args.threshold, return_visualization=bool(args.viz_out))
    if args.viz_out:
        cv2.imwrite(args.viz_out, result.pop("visualization"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
