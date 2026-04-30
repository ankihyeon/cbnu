"""
01_curate_dataset.py
====================
Roboflow YOLO segmentation 데이터셋(YOLO_data_v3)에서
균열 클래스만 포함된(crack-only) 이미지를 1,000장 추출하여
Li et al. 2024 논문 형식(8:1:1)으로 분할 저장.

원본 클래스 매핑:
  0: Crack_Crazing       (망상균열, 논문의 net-shaped)
  1: Crack_Longitudinal  (종균열,   논문의 longitudinal)
  2: Crack_Transverse    (횡균열,   논문의 transverse)
  3~7: 비균열 (human/manhole/pothole/sewer/vehicle) — 제외

산출물:
  data/curated/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/    (polygon 라벨 그대로 복사, class id 0/1/2 유지)
    ├── data.yaml                   (YOLO 학습용 설정)
    └── curation_report.json        (필터링 통계)

사용법:
    python scripts/01_curate_dataset.py \
      --src "D:/창고/vscode_AI/YOLO_data_v3" \
      --dst "data/curated" \
      --total 1000
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


CRACK_CLASSES = {0, 1, 2}  # Crack_Crazing, Longitudinal, Transverse
CLASS_NAMES = ["Crack_Crazing", "Crack_Longitudinal", "Crack_Transverse"]


def parse_label(label_path: Path) -> set[int]:
    """라벨 파일에서 등장하는 class id 집합 반환."""
    classes = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cls_id = int(line.split()[0])
            classes.add(cls_id)
        except (ValueError, IndexError):
            continue
    return classes


def is_crack_only(label_path: Path) -> tuple[bool, set[int]]:
    """
    Returns:
        (crack_only: bool, present_crack_classes: set)
        - crack_only: 균열 클래스만 있고 다른 클래스 없음
        - present_crack_classes: 등장한 균열 클래스 (0/1/2 중)
    """
    classes = parse_label(label_path)
    if not classes:
        return False, set()
    cracks = classes & CRACK_CLASSES
    others = classes - CRACK_CLASSES
    return (len(cracks) > 0 and len(others) == 0), cracks


def collect_crack_only(src_root: Path) -> list[dict]:
    """모든 split에서 crack-only 이미지 수집."""
    candidates = []
    for split in ("train", "valid", "test"):
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = lbl_dir / (img.stem + ".txt")
            crack_only, cracks = is_crack_only(label)
            if crack_only:
                # 대표 균열 유형 (다중일 때는 정렬 후 첫 항목)
                primary = sorted(cracks)[0]
                candidates.append({
                    "image": img,
                    "label": label,
                    "split_origin": split,
                    "primary_class": primary,
                    "all_cracks": sorted(cracks),
                })
    return candidates


def stratified_sample(candidates: list[dict], total: int) -> list[dict]:
    """균열 유형(primary_class) 균등 분포로 추출."""
    by_class = {0: [], 1: [], 2: []}
    for c in candidates:
        by_class[c["primary_class"]].append(c)

    per_class = total // 3
    remainder = total - per_class * 3

    rng = random.Random(42)
    sampled = []
    for cls_id, items in by_class.items():
        rng.shuffle(items)
        take = per_class + (1 if cls_id < remainder else 0)
        take = min(take, len(items))
        sampled.extend(items[:take])
        if take < (per_class + (1 if cls_id < remainder else 0)):
            print(f"[WARN] class {cls_id} ({CLASS_NAMES[cls_id]}): "
                  f"only {len(items)} available, took {take}")

    rng.shuffle(sampled)
    return sampled


def split_8_1_1(samples: list[dict]) -> dict[str, list[dict]]:
    """Li et al. 2024와 동일한 8:1:1 분할."""
    n = len(samples)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    n_train = n - n_test - n_val
    return {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }


def copy_to_curated(splits: dict[str, list[dict]], dst_root: Path) -> dict:
    """파일 복사 + 통계 산출."""
    stats = {}
    for split_name, items in splits.items():
        img_dst = dst_root / "images" / split_name
        lbl_dst = dst_root / "labels" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        class_counter = Counter()
        for item in items:
            shutil.copy2(item["image"], img_dst / item["image"].name)
            shutil.copy2(item["label"], lbl_dst / item["label"].name)
            class_counter[item["primary_class"]] += 1

        stats[split_name] = {
            "count": len(items),
            "by_class": {
                CLASS_NAMES[k]: class_counter[k] for k in (0, 1, 2)
            }
        }
    return stats


def write_data_yaml(dst_root: Path) -> None:
    """YOLO 학습용 data.yaml 생성 (서버에서 사용 시 경로 수정 필요)."""
    yaml_content = (
        "# Curated crack-only dataset for Li et al. 2024 reproduction\n"
        "# Generated by 01_curate_dataset.py\n"
        f"path: {dst_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 3\n"
        "names:\n"
        "  0: Crack_Crazing\n"
        "  1: Crack_Longitudinal\n"
        "  2: Crack_Transverse\n"
    )
    (dst_root / "data.yaml").write_text(yaml_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="원본 데이터셋 루트")
    parser.add_argument("--dst", required=True, help="큐레이션 결과 폴더")
    parser.add_argument("--total", type=int, default=1000, help="총 추출 개수")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise SystemExit(f"[ERR] 원본 데이터셋 경로 없음: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 원본 데이터셋 스캔: {src}")
    candidates = collect_crack_only(src)
    print(f"  → crack-only 후보 총 {len(candidates)}장")
    cls_dist = Counter(c["primary_class"] for c in candidates)
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"    - {name}: {cls_dist[cls_id]}장")

    if len(candidates) < args.total:
        print(f"[WARN] 후보({len(candidates)}) < 요청({args.total}), 전체 사용")
        args.total = len(candidates)

    print(f"\n[2/4] 균등 분포 추출 (총 {args.total}장)")
    sampled = stratified_sample(candidates, args.total)

    print(f"\n[3/4] 8:1:1 분할")
    splits = split_8_1_1(sampled)
    for name, items in splits.items():
        print(f"  - {name}: {len(items)}장")

    print(f"\n[4/4] 큐레이션 폴더로 복사: {dst}")
    stats = copy_to_curated(splits, dst)
    write_data_yaml(dst)

    report = {
        "source": str(src),
        "destination": str(dst.resolve()),
        "total_candidates": len(candidates),
        "selected": args.total,
        "splits": stats,
        "class_names": CLASS_NAMES,
    }
    (dst / "curation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] 완료. 보고서: {dst / 'curation_report.json'}")


if __name__ == "__main__":
    main()
