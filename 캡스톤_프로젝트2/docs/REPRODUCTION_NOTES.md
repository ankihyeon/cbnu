# 구현 노트 (Implementation Notes)

> 본 문서는 발표자료(#2) 프레임워크 중 **실데이터로 구현 가능한 모듈**을 구현하면서의 기술적 결정 사항과, 본 환경에서 발견된 데이터 특성·한계를 기록한다.

---

## 1. 구현 범위 결정

발표자료(#2)는 6개 모듈 프레임워크의 **설계 및 검증 제안** 단계로, 슬라이드 10이 "예상 산출물 / 분석 계획"이다. 즉 GPS 태깅 주행영상과 전문가/PMS 정답 label은 아직 수집 전이다.

따라서 본 저장소는 다음 원칙으로 구현 범위를 한정한다.

- **구현(실데이터 보유)**: ② 파손 탐지·분할(YOLO26n-seg), ③ 파손 정량화, ③' 도로 구간 레지스트리
- **설계로 고정(데이터 미확보)**: ④ GPS 구간 매핑, ⑤ 구간 Feature 구조화, ⑥ PCI 산출·상태평가, 검증 → `FRAMEWORK_DESIGN.md`

이는 데이터가 없는 단계를 임의 합성으로 채우지 않고, 인터페이스만 고정해 두는 방식이다.

---

## 2. 모듈별 구현 디테일

### 2.1 YOLO26n-seg 학습 — `scripts/01_train_yolo26n_seg.py`

- **모델 고정**: `yolo26n-seg.pt` (n 모델). 회사 GN-RAD 시스템 호환성과 학습 효율(경량) 고려.
- **클래스**: 데이터셋 8클래스 전체 학습. 정량화·상태평가에서는 파손 클래스(균열 3종 + 포트홀)만 사용.
- 증강은 다환경 주행영상(주·야간/맑음·우천) 대응을 위해 HSV·flip·mosaic 적용.

### 2.2 파손 정량화 — `src/damage_quantify.py`

- 클래스별 mask → connected component 분석 → 면적·개수·면적비.
- **균열**: medial-axis skeleton 길이(8-이웃 Euclidean 보정) + 거리변환 기반 평균 폭.
- **포트홀**: 면적 기반 등가지름(circle-equivalent diameter).
- **심각도 가중 점수**: `Σ severity[class] × area_ratio`. 포트홀 severity 2.5로 균열(0.8~1.0)보다 높게.
- 정밀 균열 길이·폭이 필요하면 프로젝트 #1의 BG / Optimized OrthoBoundary 모듈을 연결(본 모듈은 경량 근사).

### 2.3 도로 구간 레지스트리 — `src/road_registry.py`

- `유성구 도로.xlsx`(헤더 3번째 행)를 pandas로 로드 → `RoadSegment`(frozen dataclass) 리스트.
- `segment_id`는 도로번호(시설물명) 기반. GPS 매핑 시 spatial join 키로 사용.

---

## 3. 본 환경 데이터 특성

### 3.1 YOLO 학습 데이터셋 (`D:/YOLO_data_v3`)

| split | images | labels |
|---|---|---|
| train | 24,886 | 24,886 |
| valid | 377 | 377 |
| test | 186 | 186 |

- 8클래스: `Crack_Crazing, Crack_Longitudinal, Crack_Transverse, human, manhole, pothole, sewer, vehicle`
- Roboflow YOLO seg 포맷 (정규화 polygon)

### 3.2 유성구 도로 구간 마스터 (`유성구 도로.xlsx`)

`scripts/03_build_road_registry.py` 실행 결과 (`results/road_registry_report.json`):

| 항목 | 값 |
|---|---|
| 전체 시설 행 | 2,805 |
| 길이 집계 구간 | 2,788 |
| 총 연장 | 약 1,147.8 km |
| 평균 구간 길이 | 0.41 km |
| 최소 / 최대 | 0.006 / 410.77 km |
| 주 구분 | 시/군/구도로 2,679, 자전거도로 107 |

**데이터 품질 이슈**:
- 기준값 단위가 `km`(2,541) / `㎞`(U+339E, 140) / `m`(107) / `㎡`(5) / `대`(11) / `개`(1)로 혼재. `㎞`는 일반 `km`와 유니코드가 달라 별도 처리.
- 길이 집계는 길이 단위(km·㎞·m)만 km 환산, 면적·개수 단위(㎡·대·개) 17행은 제외.
- 최대값 410.77 km는 `구득(상.하수도시설)` 행의 입력 오류로 추정(실제 도로 구간 아님). 향후 시/군/구도로(`category == 시/군/구도로`)만 필터링하여 분석 권장.
- `도로시작위치`/`도로종료위치`는 지번 텍스트로, GPS 좌표는 없음 → 구간 geometry는 도시계획시설 도로번호 DWG/DXF에서 별도 추출 필요(④ 설계).

### 3.3 유성구청 포트홀 사진

`24·25·26년 포트홀 사진.hwp` — 한글 문서 내 임베드 이미지로, 위치정보·연도 메타만 있고 좌표 라벨은 없음. 학습/검증 활용 시 이미지 추출 + 수동 라벨링 필요.

---

## 4. 실행 검증 결과

### 4.1 smoke test (GT 기반, 가중치 불필요)

`scripts/smoke_test.py` (test split, GT polygon):

- 검출 파손 클래스: `pothole`
- 파손 면적비: 0.0199, 파손 개수: 1, 심각도 가중점수: 0.0499

→ polygon→mask→정량화 파이프라인이 끝까지 정상 동작.

### 4.2 저사양 subset 학습 (파이프라인 end-to-end 검증)

전체 데이터셋(24,886장)은 본 로컬(GTX 1050 2GB)에서 학습 불가하므로,
`scripts/00_curate_subset.py`로 소규모 subset을 추출해 **실제 학습→추론→정량화**까지 검증했다 (`results/training_summary.json`).

| 항목 | 값 |
|---|---|
| subset | train 300 / val 60 / test 60 (8클래스 stratified) |
| 학습 | yolo26n-seg, epochs 20, imgsz 320, batch 2, workers 2, cache off |
| 환경 | GTX 1050 2GB (CUDA), AMP on, **GPU mem 0.3GB**, 14.5분 |
| val Box | mAP50 0.123, mAP50-95 0.087 |
| val Mask | mAP50 0.113, mAP50-95 0.059 |
| 추론+정량화 | test 60장, conf 0.05 → pothole 검출, 면적비·심각도 산출 정상 |

**해석**: 소규모(300장·20ep·320px)라 mAP·recall이 낮아 검출이 희소하다. 이는 **절대 성능이 아니라 파이프라인 동작 검증** 결과이며, 실성능은 전체 데이터셋·서버(RTX 3090Ti) 학습에서 산출한다.

**저사양 GPU 학습 팁** (본 실행에서 확인):
- yolo26n은 3.13M params로 매우 작아 imgsz 320·batch 2면 학습 메모리 0.3GB만 사용 → 2GB VRAM에도 충분.
- 단, GPU 전용 VRAM이 데스크톱 앱들로 1.5GB 점유되어 있어, 공유 메모리 폴백 여유(시스템 RAM)가 필요. RAM 확보(Docker 등 종료)가 도움.
- `cache=False`, `workers=2`로 RAM 사용 최소화.

---

## 5. 실행 환경 (구현·검증 시점)

| 항목 | 값 |
|---|---|
| OS | Windows 11 |
| Python | 3.11 (Anaconda) |
| 주요 패키지 | numpy 2.2, opencv 4.10, scikit-image 0.25, pandas 2.2, openpyxl 3.1 |
| 탐지 | ultralytics (YOLO26 학습은 8.4+ 필요) |
| 상태평가(예정) | xgboost / lightgbm |

> 학습은 RTX 3090 Ti 서버, 정량화·레지스트리는 로컬에서 수행.

---

## 6. 다음 단계 (석사 논문 단계)

| 영역 | 현재(#2 제출) | 다음 |
|---|---|---|
| 탐지 | YOLO26n-seg 8클래스 학습 코드 | 다환경 주행영상 재학습·경량화 |
| 정량화 | medial-axis 근사 | BG/OB 정밀 측정 연계 |
| 구간 매핑 | 레지스트리 로더 | GPS·DXF 기반 map matching 구현 |
| 상태평가 | (설계) | 전문가/PMS label 확보 후 XGBoost/LightGBM |
| 검증 | (설계) | Spearman·ICC·CV·AHP·민감도 |
