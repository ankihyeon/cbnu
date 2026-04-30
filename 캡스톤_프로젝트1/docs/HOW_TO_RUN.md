# 실행 가이드 (How to Run)

> 본 문서는 본 저장소 코드를 처음 실행하는 사용자를 위한 단계별 가이드.

---

## 사전 준비

### 환경 설정

```bash
# 1) Anaconda 환경 생성
conda create -n crack-quant python=3.10 -y
conda activate crack-quant

# 2) PyTorch 설치 (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 3) 나머지 의존성
pip install -r requirements.txt

# 4) 설치 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

### 데이터셋 준비

본 저장소는 데이터를 포함하지 않으므로 다음 중 하나의 데이터를 별도 준비:

**옵션 1: 자체 YOLO segmentation 데이터셋**
- 형식: Roboflow YOLO seg (각 라벨 `class_id x1 y1 x2 y2 ... xn yn` 정규화 polygon)
- 구조: `<root>/{train,valid,test}/{images,labels}/`
- 클래스: 균열 3종(Crack_Crazing/Longitudinal/Transverse) 포함 권장

**옵션 2: AIHub 도로 균열 공개 데이터**
- AIHub "도로장애물·표면 인지 영상" 등 활용 가능
- 라벨 형식 변환 필요 (별도 스크립트 작성)

---

## 단계별 실행

### Step 1 — 데이터 큐레이션

```bash
python scripts/01_curate_dataset.py \
  --src "<원본 데이터셋 경로>" \
  --dst "data/curated" \
  --total 1000
```

**출력 확인**:
```
[1/4] 원본 데이터셋 스캔: <원본 경로>
  → crack-only 후보 총 N장
    - Crack_Crazing: ?장
    - Crack_Longitudinal: ?장
    - Crack_Transverse: ?장

[2/4] 균등 분포 추출 (총 1000장)
[3/4] 8:1:1 분할
  - train: 800장
  - val: 100장
  - test: 100장

[4/4] 큐레이션 폴더로 복사: data/curated
[OK] 완료. 보고서: data/curated/curation_report.json
```

### Step 2 — 알고리즘 동작 검증 (smoke test)

```bash
python scripts/smoke_test.py
```

**기대 출력**:
- `results/smoke_test/` 폴더에 시각화 결과 3장 저장
- 콘솔에 BG 길이 / OB 폭 측정값 출력

이 단계가 실패하면 의존성·경로 문제이므로 학습 진행 전 반드시 해결.

### Step 3 — YOLOv26n-seg 학습

```bash
# data_server.yaml의 path를 실제 절대 경로로 수정 후
python scripts/02_train_yolo26n_seg.py
```

**모니터링**:
- 학습 중 `runs/yolo26n_seg_crack/` 폴더 생성
- 에폭마다 mAP·loss 출력
- 완료 시 `runs/yolo26n_seg_crack/weights/best.pt` 생성

**참고**: GTX 1050(2GB) 등 저용량 GPU에서는 batch 4 이하, imgsz 320 등으로 축소 권장. 또는 Colab/클라우드 GPU 활용.

### Step 4 — BG/OB 평가

```bash
python scripts/03_evaluate_bg_ob.py \
  --weights runs/yolo26n_seg_crack/weights/best.pt \
  --data_root data/curated \
  --output results \
  --split test
```

**출력**:
- `results/summary_test_v2.json` — 종합 지표
- `results/per_image_test_v2.csv` — 이미지별 결과
- `results/pred_masks/`, `results/gt_masks/` — 시각화용 마스크

---

## 결과 해석

### `summary_test_v2.json` 구조

```json
{
  "split": "test",
  "n_images": 100,
  "improvements": {
    "conf_threshold": 0.5,
    "iou_threshold": 0.5,
    "mask_postprocess": "opening + closing + min_area=50"
  },
  "per_threshold": {
    "0.075": { "length": {...}, "width": {...} },
    "0.150": { "length": {...}, "width": {...} },
    "0.250": { "length": {...}, "width": {...} },
    "0.400": { "length": {...}, "width": {...} }
  },
  "best_threshold": "0.150",
  "best_metrics": {
    "length": { "PA(%)": 0.0, "NRMSE": ..., "MAE": ..., "MBE": ..., "n": 100 },
    "width":  { "PA(%)": 12.92, ... }
  }
}
```

### 평가 지표 해석

| 지표 | 정의 | 범위 |
|---|---|---|
| **PA** | (1 − MAE/평균) × 100% | 0~100% (높을수록 좋음) |
| **NRMSE** | RMSE / 평균값 | 0~∞ (낮을수록 좋음) |
| **MAE** | 평균 절대 오차 | 0~∞ (낮을수록 좋음) |
| **MBE** | 평균 편향 (양수=과추정) | -∞~+∞ (0에 가까울수록 좋음) |

---

## 자주 묻는 질문

### Q1. cv2.imread가 한국어 경로에서 실패합니다.

→ `src/io_utils.py`의 `imread_unicode()` / `imwrite_unicode()` 사용.
```python
from src.io_utils import imread_unicode, imwrite_unicode
img = imread_unicode(path)
imwrite_unicode(path, img)
```

### Q2. YOLOv26-seg 가중치가 자동 다운로드되지 않습니다.

→ 다음을 확인:
- 인터넷 연결 가능한지
- Ultralytics 8.4.0 이상인지: `pip install -U ultralytics`
- 수동 다운로드: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt

### Q3. 학습 중 OOM (Out of Memory) 에러

→ `scripts/02_train_yolo26n_seg.py`에서 다음 파라미터 조정:
```python
batch=4,        # 16 → 4
imgsz=320,      # 640 → 320
workers=2,      # 4 → 2
```

### Q4. CUDA 사용 안 됨 (CPU만)

→ `device="cpu"` 옵션으로 학습 가능하나 매우 느림. Colab 등 GPU 환경 활용 권장.

---

## 디버깅 팁

### 알고리즘 동작 단위 테스트

```python
# Python REPL 또는 새 스크립트에서
import sys
sys.path.insert(0, '.')

from src.bg_algorithm import compute_crack_length
from src.orthoboundary import compute_crack_widths
from src.io_utils import imread_unicode
import cv2

mask = imread_unicode("path/to/binary_mask.png", cv2.IMREAD_GRAYSCALE)
result = compute_crack_length(mask, threshold=0.150, return_visualization=True)
print(result)
```

### 학습 곡선 확인

```bash
# 학습 후 결과 폴더에 저장됨
results/training_curves.png       # loss / mAP 곡선
results/confusion_matrix.png      # 클래스별 혼동 행렬
```

문제 발생 시 위 그래프를 먼저 확인.
