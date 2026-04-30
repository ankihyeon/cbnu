"""
02_train_yolo26n_seg.py
=======================
YOLOv26n-seg 학습 (서버 RTX 3090 Ti 환경 가정).

논문(Li et al. 2024) 학습 설정 부분 차용:
  - Batch size 2 → 3090 Ti 24GB이므로 16~32 가능
  - Max iterations 20,000 → epochs 100 (≈20K iter at batch 16)
  - imgsz 640 (논문은 1024×512, 본 데이터셋 평균과 호환)

서버 실행 (cmd 또는 plink):
    cd D:\claude_work\paper_reproduction
    C:\ProgramData\Anaconda3\envs\roadlcc-gpu\python.exe scripts\02_train_yolo26n_seg.py
"""

from pathlib import Path

from ultralytics import YOLO


def main():
    # 작업 디렉토리는 paper_reproduction 루트라고 가정
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "configs" / "data_server.yaml"
    runs_dir = project_root / "runs"

    if not data_yaml.exists():
        raise SystemExit(f"data.yaml 없음: {data_yaml}")

    # YOLOv26n-seg 사전학습 가중치 (자동 다운로드)
    model = YOLO("yolo26n-seg.pt")

    print("=" * 60)
    print("YOLOv26n-seg 학습 시작")
    print(f"  data: {data_yaml}")
    print(f"  output: {runs_dir}")
    print("=" * 60)

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,            # 3090 Ti 24GB → 16~32 가능
        workers=4,
        device=0,
        project=str(runs_dir),
        name="yolo26n_seg_crack",
        exist_ok=True,

        # 데이터 증강 (논문 Table 5 차용)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,           # Random Scaling 50~200% 근사
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,          # Random Horizontal Flipping
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # 학습 설정
        optimizer="auto",    # YOLOv26 권장 (AdamW or SGD 자동)
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=False,        # Polynomial decay 유사 (cosine 대신 step)

        # 검증
        val=True,
        save=True,
        save_period=10,
        patience=30,
        plots=True,
    )

    print("\n[OK] 학습 완료")
    print(f"결과 저장: {runs_dir / 'yolo26n_seg_crack'}")


if __name__ == "__main__":
    main()
