# 지능화캡스톤프로젝트 #2 — 발표평가 (프레임워크 설계·검증)

> **충북대학교 산업인공지능학과 / 2026학년도 1학기**
> 과목: 지능화캡스톤프로젝트 / 평가: 프로젝트 #2 발표평가

| 항목 | 내용 |
|---|---|
| 학번 / 이름 | 2025254019 / 안기현 |
| 이메일 | ankihyeon@chungbuk.ac.kr |
| 발표 자료 | `지능화캡스톤프로젝트 양식(#2)_2025254019.pdf` (별도 제출) |

---

## 📑 과제 개요

### 연구 주제

> **지능형 도로포장관리시스템 지원을 위한 주행영상 기반 도로상태평가 프레임워크 설계 및 검증**

지방자치단체의 도로 유지관리는 시민 안전과 예산의 효율적 집행을 동시에 요구하나, 구 단위 지자체는 예산·인력·장비 제약으로 전 구간 정밀 상태조사를 반복 수행하기 어렵다. 기존 도로포장관리시스템(PMS)은 신뢰성이 높지만 정밀 장비·전문 인력·장기 이력 데이터를 전제로 해 지자체 상시 운영에 부담이 크다.

본 연구는 차량 주행영상과 딥러닝 기반 영상 분석으로, **이미지·프레임 단위 파손 인식 결과를 도로 구간 단위 평가 지표와 포장상태지수(PCI)로 연결하는 의사결정 지원형 프레임워크**를 제안한다.

```
주행영상 → 파손 인식 → GPS 기반 구간 매핑 → 정량 지표 구조화 → 포장상태지수 산출·검증
```

### 차별점

전체 PMS 구축이 아니라, **지능형 도로포장관리시스템에 활용 가능한 주행영상 기반 도로상태평가 프레임워크**의 설계·검증에 초점. 기존 연구가 파손 탐지 또는 이미지 단위 정량화에 머문 것과 달리, 파손 인식 결과를 **도로 구간 단위 평가 지표와 포장상태지수로 연결**한다.

---

## 🎯 본 저장소의 범위

발표자료(#2)는 6개 모듈 프레임워크의 **설계 및 검증 제안** 단계이며, 일부 데이터(GPS 태깅 주행영상, 전문가/PMS 정답 label)는 아직 수집 전이다. 따라서 본 저장소는 **실데이터로 동작 가능한 모듈**을 구현하고, 나머지는 인터페이스를 고정한 **설계 문서**로 정리한다.

| 모듈 | 상태 |
|---|---|
| ② 도로 파손 탐지·분할 (YOLO26n-seg, 8클래스) | ✅ 구현 |
| ③ 파손 정량화 (면적·개수·균열 길이/폭·심각도) | ✅ 구현 |
| ③' 도로 구간 레지스트리 (유성구 도로 2,805행) | ✅ 구현 |
| ④ GPS 구간 매핑 / ⑤ 구간 Feature / ⑥ PCI·상태평가 / 검증 | ⏳ 설계 (`docs/FRAMEWORK_DESIGN.md`) |

> 전체 프레임워크 설계는 [docs/FRAMEWORK_DESIGN.md](docs/FRAMEWORK_DESIGN.md) 참조.

---

## 📁 저장소 구조

```
캡스톤_프로젝트2/
├── README.md                       # 본 문서
├── requirements.txt                # Python 의존성
├── .gitignore                      # 데이터·가중치·기관자료 제외
│
├── src/                            # 핵심 구현
│   ├── __init__.py
│   ├── io_utils.py                 # 한국어 경로 안전 imread/imwrite
│   ├── polygon_to_mask.py          # YOLO seg polygon → 클래스별 mask (8클래스)
│   ├── damage_quantify.py          # 파손 정량화 (면적·개수·길이·폭·심각도)
│   └── road_registry.py            # 유성구 도로 구간 마스터 로더
│
├── scripts/
│   ├── 01_train_yolo26n_seg.py     # YOLO26n-seg 학습 (8클래스)
│   ├── 02_infer_quantify.py        # 추론 + 이미지 단위 파손 정량화
│   ├── 03_build_road_registry.py   # 도로 xlsx → 구간 레지스트리
│   ├── smoke_test.py               # GT 라벨 기반 정량화 검증 (가중치 불필요)
│   └── run_train_server.cmd        # 서버 학습 러너
│
├── configs/
│   ├── data_yolo26n.yaml           # 8클래스 학습 yaml
│   └── damage_quantify.yaml        # 정량화·심각도 파라미터
│
├── results/                        # 산출물
│   ├── road_registry_report.json   # 도로 레지스트리 요약 통계
│   └── smoke_test/smoke_result.json
│
└── docs/
    ├── FRAMEWORK_DESIGN.md          # 6개 모듈 전체 설계
    ├── HOW_TO_RUN.md               # 단계별 실행 가이드
    └── REPRODUCTION_NOTES.md        # 구현 디테일·데이터 특성·한계
```

> ⚠️ **데이터 미포함**: YOLO 학습 데이터셋 및 유성구청 원본 자료(도로 대장·CAD·포트홀 사진)는 회사·기관 보안 정책상 비공개. 코드만 제출.

---

## ⚙️ 환경 설정

```bash
conda create -n road-eval python=3.10 -y
conda activate road-eval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

| 항목 | 버전 |
|---|---|
| Python | 3.10~3.11 (Anaconda) |
| 탐지 | ultralytics ≥ 8.4 (YOLO26 지원) |
| 정량화 | opencv ≥ 4.10, scikit-image ≥ 0.25, numpy ≥ 1.26 |
| 레지스트리 | pandas ≥ 2.2, openpyxl ≥ 3.1 |
| GPU(학습) | NVIDIA RTX 3090 Ti 권장 |

---

## 🚀 실행 순서

자세한 내용은 [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) 참조.

```bash
# 0) 알고리즘 동작 검증 (가중치 불필요)
python scripts/smoke_test.py --data_root D:/YOLO_data_v3 --split test

# 1) 도로 구간 레지스트리 생성
python scripts/03_build_road_registry.py --xlsx "<유성구 도로.xlsx>" --output results

# 2) YOLO26n-seg 학습 (GPU)
python scripts/01_train_yolo26n_seg.py --data configs/data_yolo26n.yaml

# 3) 추론 + 파손 정량화
python scripts/02_infer_quantify.py \
  --weights runs/yolo26n_seg_uiseong/weights/best.pt \
  --images D:/YOLO_data_v3/test/images --output results
```

---

## 📊 구현 결과 (실데이터)

### 도로 구간 레지스트리 (`유성구 도로.xlsx`)

| 항목 | 값 |
|---|---|
| 전체 시설 행 | 2,805 |
| 길이 집계 구간 | 2,788 |
| 총 연장 | 약 1,147.8 km |
| 평균 구간 길이 | 0.41 km |
| 주 구분 | 시/군/구도로 2,679 · 자전거도로 107 |

> 기준값 단위가 km/㎞/m/㎡/대/개로 혼재되어, 길이 단위만 km 환산·집계 (데이터 품질 처리는 `docs/REPRODUCTION_NOTES.md` §3.2).

### YOLO 학습 데이터셋 (`D:/YOLO_data_v3`)

8클래스 / train 24,886 · valid 377 · test 186 (Roboflow YOLO seg).

### YOLO26n-seg 학습 (저사양 subset, 파이프라인 검증)

전체 학습은 GPU 서버에서 수행 예정이나, 본 환경(GTX 1050 2GB)에서 **소규모 subset으로 end-to-end 동작을 실제로 검증**하였다 ([results/training_summary.json](results/training_summary.json)).

| 항목 | 값 |
|---|---|
| 모델 | `yolo26n-seg.pt` (3.13M params) |
| subset | train 300 / val 60 / test 60 (8클래스, stratified) |
| 학습 | epochs 20, imgsz 320, batch 2, GTX 1050 / **14.5분**, GPU mem 0.3GB |
| val Box | P 0.584 · R 0.139 · **mAP50 0.123** · mAP50-95 0.087 |
| val Mask | P 0.603 · R 0.104 · **mAP50 0.113** · mAP50-95 0.059 |

> ⚠️ 소규모(300장·20ep·320px)라 mAP가 낮다. **절대 성능이 아닌 파이프라인 검증** 목적이며, 실성능은 전체 데이터셋·서버 학습(`01_train_yolo26n_seg.py`)에서 산출한다. 학습 곡선·혼동행렬은 [results/training_curves.png](results/training_curves.png), [results/confusion_matrix.png](results/confusion_matrix.png).

### 파손 정량화

- **GT 기반(smoke test)**: polygon→mask→정량화 정상 동작 (pothole, 면적비 0.0199, 심각도 0.0499).
- **학습 모델 기반(추론)**: 위 best.pt로 test 60장 추론 → 정량화 파이프라인 정상 작동 (conf 0.05, pothole 검출 → 면적비·심각도 산출). 결과: [results/per_image_damage.csv](results/per_image_damage.csv), [results/damage_summary.json](results/damage_summary.json).

---

## ⚠️ 한계 및 향후 방향

발표자료 슬라이드 11·12 기준 주요 한계와 개선 방향:

| 한계 | 개선 방향 |
|---|---|
| GPS 위치 오차 (도심) | Map matching·구간 단위 집계로 완화 |
| 야간·우천·모션블러 | 다환경 데이터 수집·품질 기준 적용 |
| GT/Label 확보 제한 | 일부 구간 정밀 라벨링 + 전문가 평가 병행 |
| 가중치 주관성 | AHP·상관분석·민감도·Feature Importance 비교 |
| 모델·임계값 민감성 | Cross-validation·하이퍼파라미터 튜닝 |
| 지역 편향·일반화 | 다지역·다환경 데이터 기반 일반화 검증 |

**결론**: 기존 정밀 PMS를 완전 대체하기보다, 지자체가 저비용·반복관측 기반으로 도로 구간별 상태 변화를 지속 파악할 수 있는 **보완적 도로상태평가 방법**을 목표로 한다. 향후 유지보수 이력·비용 정보와 연계 시 PMS·LCC 분석의 입력자료로 확장 가능.

---

## 📚 참고 문헌

1. 문성호, 이현종, 박동영, 한대성. (2008). 시단위 포장도로의 포장평가지수개발. *한국도로학회논문집*, 10(3), 221–230.
2. 박정권, 김창학, 최승현, 도명식. (2022). 딥러닝 기반의 도로자산 모니터링 시스템을 활용한 아스팔트 도로포장 균열률 파손모델 개발. *한국ITS학회 논문지*, 21(5), 133–148.
3. 권영주, 문성호. (2023). 드론 촬영 이미지 데이터를 기반으로 한 도로 균열 탐지 딥러닝 모델 개발. *토지주택연구*, 14(2), 125–135.
4. 박주영, 이희순, 강경태, 김병회. (2018). 동영상 분석을 통한 실시간 포장 손상 탐지 및 알림 서비스. *정보과학회 컴퓨팅의 실제 논문지*, 24(2), 59–66.
5. 변시우. (2025). 위험한 도로 노면 및 시설물 회피를 위한 딥러닝 기반 검출기. *한국산학기술학회논문지*, 26(10), 36–42.
6. Li, Z., Zhang, T., Miao, Y., Zhang, J., Torbaghan, M. E., He, Y., & Dai, J. (2024). Automated quantification of crack length and width in asphalt pavements. *Computer-Aided Civil and Infrastructure Engineering*, 39(22), 3317–3336. (프로젝트 #1 구현 대상)

---

## 📜 라이선스 및 저작권

- 본 저장소의 코드는 **학술 과제 제출용**으로, 충북대학교 산업인공지능학과 지능화캡스톤프로젝트 평가 목적으로 제출됨.
- YOLO 학습 데이터셋, 유성구청 원본 자료(도로 대장·CAD·포트홀 사진)는 회사·기관 보안 정책상 미포함.

## 📧 문의

- 안기현 (`ankihyeon@chungbuk.ac.kr`)
- 충북대학교 산업인공지능학과 / 산업인공지능연구센터
