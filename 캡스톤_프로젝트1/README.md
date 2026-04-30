# 지능화캡스톤프로젝트 #1 — 발표평가 (논문 구현)

> **충북대학교 산업인공지능학과 / 2026학년도 1학기**
> 과목: 지능화캡스톤프로젝트 / 평가: 프로젝트 #1 발표평가 (중간고사 대체)

| 항목 | 내용 |
|---|---|
| 학번 / 이름 | 2025254019 / 안기현 |
| 이메일 | ankihyeon@chungbuk.ac.kr |
| 제출 일자 | 2026-04-29 |
| 발표 자료 | `지능화캡스톤프로젝트 발표양식(#1발표평가)_2025254019.pdf` (별도 제출) |

---

## 📑 과제 개요

### 본 연구 (석사 논문 주제)

> **지자체 도로관리 한계 극복을 위한 딥러닝 기반 주행영상 분석 및 도로 포장상태지수 예측 기법**

지방자치단체의 도로 유지관리는 시민 안전 확보와 예산의 효율적 집행을 동시에 요구하는 핵심 행정 분야이나, 구 단위 지자체의 경우 예산 및 인력 제약으로 인해 정기적·정밀한 상태 조사 수행에 한계가 존재한다. 본 연구는 차량 주행영상과 딥러닝 기반 객체 탐지·분할 알고리즘을 활용하여 도로 구간 단위 도로 포장상태지수(PCI, Pavement Condition Index)를 예측하는 통합 파이프라인을 구축하는 것을 목표로 한다. 객체 탐지·공간정보 연계·정량 지표 설계·검증 체계를 통합 적용한 **융합형 연구로서 도로 포장상태지수 예측 기법을 연구개발**하는 것이 본 연구의 차별점이다.

### 선정 논문 (구현 대상)

| 항목 | 내용 |
|---|---|
| 제목 | Automated quantification of crack length and width in asphalt pavements |
| 저자 | Zhe Li, Tuo Zhang, Yi Miao, Jiupeng Zhang, Mehran Eskandari Torbaghan, Yinzhang He, Jiasheng Dai |
| 저널 | Wiley / Computer-Aided Civil and Infrastructure Engineering, 39(22), 3317–3336 (2024) |
| IF / JCR | IF 9.1 / SCIE Q1 |
| DOI | [10.1111/mice.13344](https://doi.org/10.1111/mice.13344) |
| 인용횟수 | 14회 |

본 논문은 **BG (Branch Growing) 알고리즘**과 **Optimized OrthoBoundary 알고리즘**을 제안하여 아스팔트 포장 균열의 길이와 폭을 자동 정량화하는 기법을 다룬다. 본 저장소는 위 두 알고리즘을 **저자 코드 미공개** 환경에서 직접 Python으로 구현하고, 자체 데이터셋(YOLO segmentation 1,000장)에 적용·검증한 결과를 제공한다.

#### 선정 사유

- 본 연구가 산출하고자 하는 균열 점유율·단위 길이당 파손 빈도 등 정량 지표의 핵심 입력값인 균열 길이·폭 자동 측정 알고리즘(BG, Optimized OrthoBoundary) 제안 논문
- 본 논문이 도달한 이미지 단위 정량화 수준에서 본 연구의 차별점인 도로 구간 단위 집계 및 **도로 포장상태지수 예측 기법**으로 자연스럽게 확장 가능한 연구 출발점 확보
- SCIE Q1 / IF 9.1 (Wiley CACAIE) 게재 논문으로 학술적 권위 및 인용 가치 확보, BG·OrthoBoundary 알고리즘은 본 연구의 정량화 모듈로 직접 채택 가능

---

## 📁 저장소 구조

```
캡스톤_프로젝트1/
├── README.md                          # 본 문서
├── requirements.txt                   # Python 의존성
├── .gitignore                         # 데이터·가중치 제외 설정
│
├── src/                               # 알고리즘 핵심 구현
│   ├── __init__.py                    # 패키지 진입점
│   ├── bg_algorithm.py                # BG 알고리즘 (논문 §3)
│   ├── orthoboundary.py               # Optimized OrthoBoundary (논문 §4)
│   ├── polygon_to_mask.py             # YOLO seg polygon → binary mask
│   └── io_utils.py                    # 한국어 경로 안전 imread/imwrite
│
├── scripts/                           # 실행 스크립트
│   ├── 01_curate_dataset.py           # 균열 1,000장 추출 + 8:1:1 분할
│   ├── 02_train_yolo26n_seg.py        # YOLOv26n-seg 학습
│   ├── 03_evaluate_bg_ob.py           # 학습 모델 + BG/OB 평가 (3단계 보정)
│   ├── smoke_test.py                  # 알고리즘 동작 검증용
│   └── run_train_server.cmd           # SSH 서버 학습 러너
│
├── configs/                           # 설정 파일
│   ├── data_server.yaml               # 서버 학습용 yaml
│   └── data_curated.yaml              # 큐레이션 결과 yaml
│
├── results/                           # 평가·학습 결과 산출물
│   ├── summary_test.json              # 1차 평가 (개선 전)
│   ├── summary_test_v2.json           # 2차 평가 (3단계 보정 기법)
│   ├── per_image_test.csv             # 1차 이미지별 결과
│   ├── per_image_test_v2.csv          # 2차 이미지별 결과
│   ├── curation_report.json           # 데이터셋 큐레이션 통계
│   ├── training_curves.png            # 학습 손실·mAP 곡선
│   └── confusion_matrix.png           # 검증 혼동행렬
│
└── docs/                              # 추가 문서
    ├── REPRODUCTION_NOTES.md          # 구현 디테일·결정 사항·한계
    └── HOW_TO_RUN.md                  # 단계별 실행 가이드
```

> ⚠️ **데이터셋 미포함**: 본 연구 자체 데이터셋(GN-RAD 시스템 수집)은 회사 보안 정책상 비공개. 코드만 제출.

---

## ⚙️ 환경 설정

### 권장 환경

| 항목 | 버전 |
|---|---|
| OS | Windows Server / 10 / 11 |
| Python | 3.10.x (Anaconda) |
| PyTorch | 2.5.1 + CUDA 12.1 |
| GPU | NVIDIA RTX 3090 Ti (24GB VRAM) — 학습 시 권장 |

### 설치

```bash
# 가상환경 생성
conda create -n crack-quant python=3.10 -y
conda activate crack-quant

# PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성
pip install -r requirements.txt
```

### 의존성 (`requirements.txt`)

```
ultralytics>=8.4.0      # YOLOv26 지원 필요
opencv-python>=4.10
scikit-image>=0.25
networkx>=3.4
numpy>=1.26
pyyaml>=6.0
```

---

## 🚀 실행 순서

본 연구의 전체 파이프라인은 다음 4단계로 구성된다.

### 1단계 — 데이터셋 큐레이션

자체 YOLO segmentation 데이터셋(8개 클래스)에서 **균열 클래스(Crazing/Longitudinal/Transverse)만** 포함된 이미지 1,000장을 추출하여 8:1:1로 분할.

```bash
python scripts/01_curate_dataset.py \
  --src "<원본 데이터셋 경로>" \
  --dst "data/curated" \
  --total 1000
```

**출력**: `data/curated/{images,labels}/{train,val,test}/` + `data.yaml` + `curation_report.json`

### 2단계 — 알고리즘 동작 검증

학습 없이 `data/curated`의 첫 테스트 이미지에 대해 polygon → mask → BG/OB 파이프라인을 실행하여 알고리즘이 정상 동작하는지 확인.

```bash
python scripts/smoke_test.py
```

**출력**: `results/smoke_test/{input_image.jpg, gt_mask.png, bg_visualization.png}` + 콘솔에 길이·폭 측정값.

### 3단계 — YOLOv26n-seg 학습 (3090Ti 권장)

균열 1,000장으로 segmentation 모델을 학습.

```bash
python scripts/02_train_yolo26n_seg.py
```

**학습 설정**:
- Model: yolo26n-seg.pt (Ultralytics v8.4+, 2.7M params, 9.0 GFLOPs)
- Epochs: 100, Batch: 16, imgsz: 640
- Optimizer: auto (AdamW/SGD)
- Augmentation: HSV, flip, mosaic
- ETA: ~50분 (RTX 3090 Ti)

**출력**: `runs/yolo26n_seg_crack/weights/best.pt`

### 4단계 — BG/OB 평가 (3단계 정량화 보정 기법 적용)

학습된 모델로 test 100장에 대해 inference 후 BG/OB 알고리즘을 적용하여 길이·폭 정량화 정확도를 산출.

```bash
python scripts/03_evaluate_bg_ob.py \
  --weights runs/yolo26n_seg_crack/weights/best.pt \
  --data_root data/curated \
  --output results \
  --split test
```

**적용된 3단계 정량화 보정 기법**:
1. 신뢰도 임계값 상향 (0.25 → **0.5**) — 잡음 마스크 제거
2. 마스크 후처리 — Opening + Closing + min_area 50px 필터
3. BG threshold grid search — 0.075 / 0.150 / 0.250 / 0.400 중 최적값 자동 선정

**출력**: `results/summary_test_v2.json` + `per_image_test_v2.csv`

---

## 📊 실험 결과 요약

### Segmentation 성능 (YOLOv26n-seg, val 100장)

| 클래스 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Crack_Crazing (망상) | 0.755 | 0.527 | **0.573** ★ | 0.255 |
| Crack_Longitudinal (종) | 0.759 | 0.333 | 0.498 | 0.229 |
| Crack_Transverse (횡) | 0.695 | 0.451 | 0.493 | 0.225 |
| 전체 (all) | 0.736 | 0.437 | 0.522 | 0.236 |

→ **망상균열 1위 패턴**은 논문(Li et al. 2024)의 net-shaped > longitudinal > transverse 순위 패턴과 동일하게 재현됨.

### BG threshold Grid Search 결과 (3단계 보정 기법)

| Threshold | Length PA | Length MBE | Width PA | Width MBE |
|---|---|---|---|---|
| t = 0.075 | 0.0% | +551.8 | 11.8% | +2.91 |
| **t = 0.150** ★ | **0.0%** | **+303.2** | **12.92%** | **+2.93** |
| t = 0.250 | 0.0% | +23.5 | 12.52% | +2.79 |
| t = 0.400 | 0.0% | −174.7 | 11.20% | +3.20 |

### BG/OB 정량화 결과 (test 100장, 1차 vs 2차 vs 선정 논문)

#### 길이 측정 (BG 알고리즘)

| 구분 | Length PA | Length MAE | Length MBE |
|---|---|---|---|
| 1차 (개선 전) | 0.0% | 1593.0 | +1388.8 |
| **2차 (3단계 보정, t=0.150) ★** | **0.0%** | **1370.5** | **+303.2** |
| 선정 논문 | 82.71% | 667.7 | +404.4 |

#### 폭 측정 (Optimized OrthoBoundary)

| 구분 | Width PA | Width MAE | Width MBE |
|---|---|---|---|
| 1차 (개선 전) | 24.72% | 14.89 | +12.40 |
| **2차 (3단계 보정, t=0.150) ★** | **12.92%** | **17.21** | **+2.93** |
| 선정 논문 | 85.55% | 4.49 | −0.029 |

### 핵심 발견

1. **편향(MBE) 76% 이상 감소** — 3단계 정량화 보정 기법 적용으로 over-segmentation 거의 해소
2. **절대 PA는 논문 미달** — 모델 용량(yolo26n 2.7M params) 한계 + GT 형식 차이(polygon vs 캘리퍼)가 핵심 원인
3. **망상균열 우위 패턴 재현** — Crazing > Longitudinal > Transverse 순위가 본 환경에서도 동일

---

## ⚠️ 본 연구와의 대비를 통한 선정 논문의 한계점

| # | 한계 영역 | 내용 |
|---|---|---|
| ① | **데이터 환경의 근본적 차이** | 고정 삼각대 기반 정적 근접 촬영(3024×3024) 환경 가정하에 알고리즘 설계되었으나, 본 연구의 차량 주행(50~80km/h) 환경 및 모션 블러 발생 시 직접 적용 가능성 검증 부재 |
| ② | **측정 단위 한계 ★** | 이미지 단위 균열 길이·폭 측정에 그쳐 도로 구간(교차로~교차로) 단위 집계 및 **도로 포장상태지수 설계 미수행**으로 도로 유지관리 행정 단위와 미스매치 발생 |
| ③ | **공간정보(GPS) 연계 부재** | GPS 등 위치정보 결합 미수행으로 인해 동일 균열의 반복 관측 및 시계열 분석 불가능, 본 연구의 위치 기반 구간 매핑 단계 부재 |
| ④ | **알고리즘 적용 한계** | Single threshold 0.075 일괄 적용으로 인한 일부 균열의 critical feature 손실 발생, 횡균열 정확도가 종/망상 균열 대비 낮으며 포트홀 등 면형 파손 미포함 |
| ⑤ | **재현성 한계** | 데이터셋(Birmingham 자체 수집) 및 BG/OB 알고리즘 코드 미공개로 인해 본 연구 환경 재현 시 자체 데이터(AIHub 등) 확보 필요 |
| ⑥ | **평가 GT 한계** | 캘리퍼 수동 측정 GT의 주관성 및 기기 오차 한계 존재, Model 1-BG의 MBE +404.4 양의 편향 발생, 전문가 평가 및 기존 PMS 정합성 검증 미수행 |

### 의견

- 선정 논문은 이미지 단위 균열 정량화 SOTA 알고리즘을 제시하여 본 연구의 정량화 모듈로 직접 채택 가능하며, 이미지 → 구간 단위 확장·GPS 매핑·**도로 포장상태지수 예측 기법** 영역에서 명확한 차별점 확보 가능
- 본 환경에서 직접 재현한 실험 결과, 임계값 조정 및 마스크 후처리 적용을 통해 측정 편향(MBE) 76% 이상 감소를 확인하였으나, 절대 정량화 정확도는 논문 결과 대비 격차가 발생하여 데이터 환경 차이 및 평가 GT 형식 한계가 본 연구의 핵심 개선 대상임을 실증적으로 확인

---

## 🚀 향후 연구 방향

본 연구의 후속 단계로 다음을 진행할 예정:

| 한계 | 본 연구의 해결 방향 |
|---|---|
| ① **정적 촬영 환경** | 차량 부착 스마트폰 기반 다환경(주·야간/맑음·우천) 주행영상 수집 및 모션 블러 보정·다중 프레임 융합 적용을 통한 동적 환경 대응 |
| ② **이미지 단위 측정** | [1단계] YOLO 실시간 탐지 → [2단계] FCN-HRNet 분할 + BG/OB 정량화 → GPS 매핑을 통한 구간 단위 집계 → **도로 포장상태지수 예측 통합 파이프라인 구축** |
| ③ **공간정보 부재** | GPS 및 주행 경로 매핑을 통한 도로 구간 자동 분할 수행 및 동일 구간 다회 관측 기반 시계열 안정성 분석 수행 |
| ④ **알고리즘 적용** | 도로 유형별 적응형 threshold 적용 및 균열 데이터 균형화를 통한 정확도 개선, 객체탐지·segmentation 결합을 통한 포트홀 등 면형 파손 추가 대응 |
| ⑤ **재현성** | AI-Hub 도로 균열 공개 데이터셋 및 자체 GN-RAD 시스템 데이터 활용을 통한 한국 도로 환경 재현, 전체 구현 코드·모델 GitHub 공개를 통한 오픈 사이언스 실현 |
| ⑥ **평가 기준** | 전문가 AHP 평가 및 기존 PMS 결과와의 Spearman 상관 분석 수행, 동일 구간 반복 관측 ICC 분석을 통한 안정성 검증 |

### 본 연구 최종 목표

- 본 연구는 선정 논문이 제시한 이미지 단위 균열 길이·폭 산출 기법을 출발점으로 하여, 주행영상 → 탐지·분할 → GPS 매핑 → 구간 정량지표 → **도로 포장상태지수(0~100점) 산출**에 이르는 통합 파이프라인을 구축
- 산출된 도로 포장상태지수는 전문가 평가와의 Spearman 상관계수 0.75 이상 및 동일 구간 반복 관측 ICC 0.80 이상의 안정성과 1km 주행영상당 5분 이내 처리 속도를 목표로 **유성구 관할 도로 적용 및 검증**을 통해 정합성·재현성과 실무 활용 가능성 입증
- 본 환경 재현 실험을 통해 정량화 알고리즘 통합 모듈의 작동 가능성 및 측정 편향(MBE) 76% 감소 효과를 검증하였으며, 모델 용량 확장 및 정밀 측정 기반 GT 도입을 통한 절대 정량화 정확도 향상을 후속 연구 과제로 도출
- 선정 논문의 이미지 단위 정량화에서 지자체 도로관리 의사결정 지원 체계로 확장한 융합형 연구로서 학술적·실무적 의의 확보

---

## 📚 참고 문헌

1. 문성호, 이현종, 박동영, 한대성. (2008). 시단위 포장도로의 포장평가지수개발. *한국도로학회논문집*, 10(3), 221–230.

2. 박정권, 김창학, 최승현, 도명식. (2022). 딥러닝 기반의 도로자산 모니터링 시스템을 활용한 아스팔트 도로포장 균열률 파손모델 개발. *한국ITS학회 논문지*, 21(5), 133–148. DOI: [10.12815/kits.2022.21.5.133](https://doi.org/10.12815/kits.2022.21.5.133)

3. 권영주, 문성호. (2023). 드론 촬영 이미지 데이터를 기반으로 한 도로 균열 탐지 딥러닝 모델 개발. *토지주택연구*, 14(2), 125–135. DOI: [10.5804/LHIJ.2023.14.2.125](https://doi.org/10.5804/LHIJ.2023.14.2.125)

4. 박주영, 이희순, 강경태, 김병회. (2018). 동영상 분석을 통한 실시간 포장 손상 탐지 및 알림 서비스. *정보과학회 컴퓨팅의 실제 논문지*, 24(2), 59–66. DOI: [10.5626/KTCP.2018.24.2.59](https://doi.org/10.5626/KTCP.2018.24.2.59)

5. 변시우. (2025). 위험한 도로 노면 및 시설물 회피를 위한 딥러닝 기반 검출기. *한국산학기술학회논문지*, 26(10), 36–42. DOI: [10.5762/KAIS.2025.26.10.36](https://doi.org/10.5762/KAIS.2025.26.10.36)

> **선정 논문**: Li, Z., Zhang, T., Miao, Y., Zhang, J., Torbaghan, M. E., He, Y., & Dai, J. (2024). Automated quantification of crack length and width in asphalt pavements. *Computer-Aided Civil and Infrastructure Engineering*, 39(22), 3317–3336. DOI: [10.1111/mice.13344](https://doi.org/10.1111/mice.13344)

---

## 📜 라이선스 및 저작권

- 본 저장소의 코드는 **학술 과제 제출용**으로, 충북대학교 산업인공지능학과 지능화캡스톤프로젝트 평가 목적으로 제출됨.
- 자체 데이터셋(GN-RAD 시스템) 및 학습된 모델 가중치는 회사 보안 정책상 미포함.
- 알고리즘 구현은 Li et al. (2024) 논문의 의사코드를 기반으로 직접 작성됨 (저자 코드 미공개).

---

## 📧 문의

- 안기현 (`ankihyeon@chungbuk.ac.kr`)
- 충북대학교 산업인공지능학과 / 산업인공지능연구센터
