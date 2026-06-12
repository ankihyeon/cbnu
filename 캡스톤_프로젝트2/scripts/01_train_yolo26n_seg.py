"""
01_train_yolo26n_seg.py
=======================
YOLO26n-seg 학습 (전체 8클래스, 서버 RTX 3090 Ti 환경 가정).

발표자료(#2) 모듈 ② "도로 파손 탐지·분할"의 학습 단계.
data.yaml은 8클래스(균열 3종 + human/manhole/pothole/sewer/vehicle)를
그대로 사용한다. 정량화·상태평가에서는 파손 클래스(균열·포트홀)만 활용한다.

서버 실행:
    python scripts/01_train_yolo26n_seg.py --data configs/data_yolo26n.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data_yolo26n.yaml",
                        help="YOLO data.yaml (8클래스)")
    parser.add_argument("--model", default="yolo26n-seg.pt",
                        help="사전학습 가중치 (n 모델 고정)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (저RAM 환경은 2 권장)")
    parser.add_argument("--cache", default="False",
                        help="이미지 캐시: False/ram/disk (저RAM은 False)")
    parser.add_argument("--name", default="yolo26n_seg_uiseong")
    args = parser.parse_args()

    cache = {"false": False, "true": True, "ram": "ram", "disk": "disk"}.get(
        str(args.cache).lower(), False)

    from ultralytics import YOLO

    project_root = Path(__file__).resolve().parent.parent
    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = project_root / data_yaml
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml 없음: {data_yaml}")

    runs_dir = project_root / "runs"

    model = YOLO(args.model)  # n 모델
    print("=" * 60)
    print("YOLO26n-seg 학습 시작 (8 classes)")
    print(f"  data:   {data_yaml}")
    print(f"  model:  {args.model}")
    print(f"  output: {runs_dir}")
    print("=" * 60)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=cache,
        device=args.device,
        project=str(runs_dir),
        name=args.name,
        exist_ok=True,

        # 데이터 증강 (다환경 주행영상 대응 — 발표자료 슬라이드 9 수집 조건 반영)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,

        optimizer="auto",
        lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=False,

        val=True, save=True, save_period=10, patience=30, plots=True,
    )

    print("\n[OK] 학습 완료")
    print(f"결과: {runs_dir / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
