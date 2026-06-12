# 실행 가이드 (How to Run)

> 본 저장소는 발표자료(#2) 프레임워크 중 **실데이터로 동작 가능한 모듈**(YOLO26n-seg 학습·추론, 파손 정량화, 도로 구간 레지스트리)의 실행 가이드를 제공한다. GPS 매핑·PCI 산출·상태평가·검증은 `FRAMEWORK_DESIGN.md` 참조.

---

## 사전 준비

### 환경 설정

```bash
conda create -n road-eval python=3.10 -y
conda activate road-eval

# PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성
pip install -r requirements.txt

# 설치 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

### 데이터 준비

- **YOLO 학습 데이터셋**: 8클래스 YOLO segmentation 데이터셋
  - 구조: `<root>/{train,valid,test}/{images,labels}/` + `data.yaml`
  - 클래스: `Crack_Crazing, Crack_Longitudinal, Crack_Transverse, human, manhole, pothole, sewer, vehicle`
  - 예시 경로: `D:/YOLO_data_v3`
- **도로 구간 마스터**: 지자체 도로 대장 xlsx (유성구 도로.xlsx)
  - 헤더 3번째 행, 컬럼: `NO | 시설물명 | 업종명 | 도로명주소 | 관리부서명 | 도로시작위치 | 도로종료위치 | 기준값 | 기준값단위`

> ⚠️ 데이터셋·지자체 원본 자료는 비공개이며 본 저장소에 포함되지 않는다. (`.gitignore`로 제외)

---

## 단계별 실행

### Step 0 — 알고리즘 동작 검증 (smoke test, 가중치 불필요)

GT polygon 라벨로 polygon→mask→정량화 파이프라인을 검증. **학습 없이 즉시 실행 가능**.

```bash
python scripts/smoke_test.py --data_root D:/YOLO_data_v3 --split test
```

**출력**: `results/smoke_test/`에 입력 이미지·파손 마스크·`smoke_result.json` 저장, 콘솔에 정량 지표 출력.

### Step 1 — 도로 구간 레지스트리 생성

지자체 도로 xlsx → 표준화 레지스트리(CSV) + 요약(JSON).

```bash
python scripts/03_build_road_registry.py \
  --xlsx "<유성구 도로.xlsx 경로>" \
  --output results
```

**출력**: `results/road_registry.csv`(구간 목록), `results/road_registry_report.json`(요약 통계).

> 기준값 단위가 `km`/`㎞`/`m`/`㎡`/`대`/`개`로 혼재되어 있어, 길이 집계는 길이 단위(km·㎞·m)만 km로 환산하며 그 외는 별도 카운트한다.

### Step 2 — YOLO26n-seg 학습 (GPU 권장)

```bash
# configs/data_yolo26n.yaml의 path를 데이터셋 절대경로로 수정 후
python scripts/01_train_yolo26n_seg.py --data configs/data_yolo26n.yaml
```

- Model: `yolo26n-seg.pt` (n 모델 고정)
- Epochs 100, imgsz 640, batch 16 (RTX 3090 Ti 기준)
- 출력: `runs/yolo26n_seg_uiseong/weights/best.pt`

**참고**: 저용량 GPU는 `--batch 4 --imgsz 320` 등으로 축소. CPU는 매우 느림.

### Step 3 — 추론 + 파손 정량화

```bash
python scripts/02_infer_quantify.py \
  --weights runs/yolo26n_seg_uiseong/weights/best.pt \
  --images D:/YOLO_data_v3/test/images \
  --output results
```

**출력**: `results/per_image_damage.csv`(이미지별 파손 지표), `results/damage_summary.json`.

---

## 정량 지표 설명

| 지표 | 정의 |
|---|---|
| `damage_area_ratio` | 파손 mask 면적 / 이미지 면적 (균열 점유율의 이미지 단위 기초값) |
| `damage_count` | 파손 인스턴스(컴포넌트) 수 |
| `crack_length_px` | 균열 skeleton 길이 (medial-axis 기반, px) |
| `mean_width_px` | 균열 평균 폭 (거리변환 기반, px) |
| `severity_score` | Σ (클래스 심각도 × 면적비) — 포트홀 가중 ↑ |

이미지 단위 지표 → (설계 단계) GPS 구간 매핑 → 구간 Feature → PCI로 확장.

---

## 자주 묻는 질문

### Q. cv2.imread가 한국어 경로에서 실패합니다.
→ `src/io_utils.py`의 `imread_unicode()` / `imwrite_unicode()` 사용.

### Q. yolo26n-seg.pt 자동 다운로드가 안 됩니다.
→ Ultralytics 8.4.0 이상 필요(`pip install -U ultralytics`). YOLO26 미지원 버전에서는 학습이 실패한다.

### Q. 균열 길이·폭을 더 정밀하게 측정하고 싶습니다.
→ 프로젝트 #1의 BG / Optimized OrthoBoundary 알고리즘(Li et al. 2024) 모듈을 연결한다. 본 모듈은 medial-axis 기반 경량 근사를 제공.
