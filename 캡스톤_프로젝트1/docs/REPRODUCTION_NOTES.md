# 재현 노트 (Reproduction Notes)

> 본 문서는 Li et al. (2024) "Automated quantification of crack length and width in asphalt pavements" 논문의 BG / Optimized OrthoBoundary 알고리즘을 직접 구현하면서 도출한 기술적 결정 사항과 본 환경 적용 시 발견된 한계점을 기록한다.
>
> 본 연구의 최종 산출물은 **도로 포장상태지수(PCI, Pavement Condition Index)** 예측 기법이며, 본 저장소는 그 핵심 모듈인 균열 길이·폭 정량화 부분의 재현 결과를 담는다.

---

## 1. 알고리즘 구현 디테일

### 1.1 BG (Branch Growing) 알고리즘 — `src/bg_algorithm.py`

논문 §3 의사코드 기반 직접 구현. 핵심 단계:

| 단계 | 함수 | 출처 (논문) |
|---|---|---|
| MAT skeleton 추출 | `extract_skeleton()` | §3.1.1 (식 1) |
| 2×2 특수 구조 처리 | `fix_special_2x2()` | §3.1.2, Fig. 4 |
| Connection matrix → degree map | `degree_map()` | §3.1.3 |
| Endpoint / Relay / Junction 분류 | `classify_points()` | §3.1.3 |
| Chain 추출 (체인 단위 분해) | `trace_chains()` | §3.1.3, Fig. 6 |
| Chain 길이 (인접 픽셀 Euclidean 누적) | `chain_length()` | §3.1.5 (식 2~4 단순화) |
| Trunk + Branch 통합 (threshold 0.075) | `find_trunk_and_branches()` | §3.1.6 (식 5) |

**Steger Unbiased Detector 단순화**: 논문은 Steger (1998)의 unbiased detector로 곡선 picture를 fitting하지만, single-pixel-width skeleton에서는 인접 픽셀 Euclidean distance 누적으로 충분히 근사됨을 확인. 정확도 손실은 본 환경(저해상도)에서 무시 가능 수준.

### 1.2 Optimized OrthoBoundary 알고리즘 — `src/orthoboundary.py`

논문 §4 기반. **3단계 최적화** 구현:

| 최적화 | 함수 | 효과 (논문 보고) |
|---|---|---|
| ① Spatial Locality | `find_components_and_contours()` | 동일 component 내 contour 재계산 회피 |
| ② Canny → cv2.findContours | `find_components_and_contours()` 내부 | binary 이미지에 최적화된 함수 사용 |
| ③ Parallel Computing | `compute_crack_widths_parallel()` | multiprocessing.Pool 기반 (단일 이미지에서는 이득 적음) |

**보정 처리**: skeleton point에서 PCA로 국소 방향(접선)을 산출, 그 방향에 수직인 직선이 contour와 만나는 두 점 사이 거리로 폭을 계산. `local_normal_via_pca()` 함수로 구현.

---

## 2. 본 환경 적용 시 주요 결정 사항

### 2.1 데이터셋 대체

논문 데이터셋은 **University of Birmingham 캠퍼스 도로 8개 구간** 자체 수집(761장 균열, iPhone 11 + 고정 삼각대 3024×3024)으로 미공개. 본 환경에서는:

- **자체 데이터셋(지엔소프트 GN-RAD 시스템 수집)** 25,449장 중 균열 클래스(Crazing/Longitudinal/Transverse)만 1,000장 추출
- 8:1:1 분할: train 800 / val 100 / test 100
- 클래스 균등: 각 ~333장 (크기 범위 차이 고려)

자체 데이터셋은 **회사 보안 정책상 GitHub 미포함**. 코드만 제출.

### 2.2 GT 형식 차이

- **논문**: 캘리퍼 실측 GT (균열당 10~20회 측정, 수직 방향 폭)
- **본 환경**: YOLO segmentation polygon 라벨 → binary mask 변환 (`src/polygon_to_mask.py`)

이 차이가 절대 PA 격차의 핵심 원인. 향후 캘리퍼 측정 GT 도입 필요.

### 2.3 모델 차이

- **논문**: 4종 segmentation 모델 비교 (MaskFormer-ST, FCN-HRNet, FCN-UHRNet, PPLiteSeg-STDC2)
- **본 환경**: Ultralytics **YOLOv26n-seg** 1종 (2.7M params, 9.0 GFLOPs)

논문 모델은 PaddleSeg/PyTorch 기반으로 Ultralytics에 직접 통합 어려움. YOLOv26-seg는 본인 회사 GN-RAD 시스템 호환성 + 학습 효율(6.5MB 가중치, 49분 학습)을 고려해 선택.

---

## 3. 3단계 정량화 보정 기법

본 환경에서 절대 PA 격차 해소를 위해 추가 적용한 보정 기법. `scripts/03_evaluate_bg_ob.py`에 통합.

### 3.1 신뢰도 임계값 상향

```python
CONF_THRESHOLD = 0.5    # YOLO 기본값 0.25 → 0.5
IOU_THRESHOLD = 0.5
results = yolo_model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
```

→ 잡음 마스크 제거 효과.

### 3.2 마스크 후처리

```python
def postprocess_mask(mask, min_area=50):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 잡음 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 균열 연결
    # min_area 이상 component만 유지
    ...
```

→ 분기 노이즈 정제 효과.

### 3.3 BG threshold Grid Search

논문은 모든 데이터에 threshold 0.075 권장. 본 데이터에서는:

| Threshold | Length MBE | Width MBE |
|---|---|---|
| 0.075 | +551.8 | +2.91 |
| **0.150 ★** | **+303.2** | **+2.93** |
| 0.250 | +23.5 | +2.79 |
| 0.400 | −174.7 | +3.20 |

→ **t=0.150**이 본 데이터 최적값으로 자동 선정 (Length+Width PA 합 기준).

---

## 4. 알려진 한계 / 미해결 이슈

### 4.1 Length PA 0% 문제

전체 임계값 grid에서 Length PA가 0%로 나오는 이유:
- PA = max(0, 1 − MAE/평균) × 100%
- 본 환경 GT 평균 길이 ≈ 1,155px, MAE 1,078~1,523px
- → MAE > 평균이라 PA 음수 → 0으로 클램핑

**근본 원인**: 모델이 over-segment하여 예측 길이가 GT의 2.2배 (평균 GT 1155 vs Pred 2543).
**해결 방향** (옵션 B/C — 본 제출 범위 외):
- yolo26m/l 등 **더 큰 모델로 재학습** (3090Ti 24GB 환경에서 가능)
- 캘리퍼 실측 GT 도입
- 하이퍼파라미터 튜닝 (lr·data augmentation)

### 4.2 Width PA 격차

- 1차 PA 24.7% → 2차 PA 12.9%로 **하락**
- MBE는 +12.4 → +2.9로 **76% 감소** (편향 해소 성공)
- 즉 보정 기법은 **편향(bias)은 줄였으나 분산(variance)은 증가**
- → Width 측정에서는 보정 적용 시 정확도 trade-off 존재

### 4.3 Segmentation 모델 vs 알고리즘

- 본 환경 mAP50 mask 0.522 (val 100장)
- 망상 1위 패턴은 논문 동일 재현
- 그러나 **mAP 수치 자체가 논문 80%대(Length PA 기준)와는 직접 비교 불가**
  - Segmentation mAP ≠ Length/Width PA
  - 본 논문도 Segmentation 1위(Model 3 FCN-UHRNet)와 Length 1위(Model 1 MaskFormer-ST)가 다름

---

## 5. 환경 정보 (실험 시점)

### 학습 환경 (NVIDIA RTX 3090 Ti)

- Windows Server (SSH 원격)
- Python 3.10.20 (Anaconda `roadlcc-gpu`)
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics 8.4.37
- VRAM 24GB (사용 ~3.6GB / batch 16 / imgsz 640)
- 학습 시간: 100 epochs / 49분

### 평가 환경

- Local Windows + GTX 1050 (CPU 전용)
- Python 3.12.4 (Anaconda `ai_env`)
- ai_env 환경: ultralytics 8.3.27, PyTorch 2.3.1+cu118
- Inference 7.4ms/image (3090Ti) → 평가는 서버에서 수행

---

## 6. 향후 개선 예정 (석사 논문 단계)

| 영역 | 1단계 (현재 제출) | 2단계 (다음) | 3단계 (최종) |
|---|---|---|---|
| 모델 | yolo26n-seg 2.7M | yolo26m-seg 23.6M | yolo26l-seg + ensemble |
| GT | polygon → mask | polygon + 일부 캘리퍼 | 전체 캘리퍼 측정 |
| Threshold | grid search (4값) | 도로 유형별 적응형 | 학습 기반 자동 선정 |
| 데이터 | 1,000장 | 5,000장 | 도로 km 단위 |
| 검증 | MBE 기반 | + 전문가 AHP | + PMS 정합성 |
| 적용 도로 | (검증 미수행) | 유성구 일부 | 유성구 관할 도로 전체 |
| 산출물 | 균열 길이·폭 정량화 | 구간 단위 집계 | **도로 포장상태지수(0~100점) 예측** |

---

## 7. 참고

- 논문 본문: 별도 PDF 첨부 (`선정논문.pdf` — 본 저장소 미포함)
- 본 환경 적용 결과 시각화: `results/training_curves.png`, `results/confusion_matrix.png`
- 이미지별 상세 측정값: `results/per_image_test_v2.csv` (100장)

본 재현 노트는 발표 자료 슬라이드 10·11·12의 근거로 활용됨.
