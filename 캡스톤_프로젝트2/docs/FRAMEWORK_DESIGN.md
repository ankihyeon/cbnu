# 프레임워크 설계 (Framework Design)

> 본 문서는 발표자료(#2) **"지능형 도로포장관리시스템 지원을 위한 주행영상 기반 도로상태평가 프레임워크 설계 및 검증"** 의 전체 6개 모듈 설계를 정리한다.
>
> 이 중 **실데이터로 구현·검증 가능한 모듈(②~③의 일부)** 은 본 저장소 코드로 제공하며, 나머지(GPS 구간 매핑·구간 Feature 집계·포장상태지수 산출·상태평가 모델·검증)는 **데이터 확보 후 진행 예정인 설계**로 기술한다.

---

## 전체 파이프라인

```
① 데이터 수집        주행영상(차량부착 스마트폰) + GPS 로그 + 촬영시각
        ↓
② 도로 파손 탐지·분할  YOLO26n-seg → 클래스별 mask, bbox, confidence
        ↓
③ 파손 정량화         mask 면적·개수, 균열 길이·폭(skeletonization), 심각도
        ↓
④ 공간정보 매핑        GPS 보정 → map matching → 도로 구간(교차로~교차로) spatial join
        ↓
⑤ 구간 Feature 구조화   균열점유율·파손빈도·심각도·공간집중도·반복파손지속도 등 정규화
        ↓
⑥ 포장상태지수 산출     가중치 1차 기준지수 → XGBoost/LightGBM 보정·등급 예측
        ↓
   검증               전문가/PMS 비교(Spearman) · 반복안정성(ICC·CV) · 민감도·AHP
```

---

## 구현 현황 (본 저장소)

| 모듈 | 상태 | 구현 위치 | 비고 |
|---|---|---|---|
| ① 데이터 수집 | ⏳ 설계 | — | GPS 태깅 주행영상 미수집 (회사 GN-RAD 연계 예정) |
| ② 파손 탐지·분할 | ✅ 구현 | `scripts/01_train_yolo26n_seg.py`, `02_infer_quantify.py` | YOLO26n-seg, 8클래스 학습 가능 |
| ③ 파손 정량화 | ✅ 구현 | `src/damage_quantify.py` | 면적·개수·균열 길이/폭·심각도 |
| ③' 도로 구간 레지스트리 | ✅ 구현 | `src/road_registry.py`, `scripts/03_build_road_registry.py` | 유성구 도로 2,805행 로드 |
| ④ GPS 구간 매핑 | ⏳ 설계 | — | GPS 좌표 데이터 필요 (아래 설계) |
| ⑤ 구간 Feature 구조화 | ⏳ 설계 | — | ④ 출력 의존 (아래 설계) |
| ⑥ PCI 산출 + 상태평가 | ⏳ 설계 | — | 전문가/PMS label 필요 (아래 설계) |
| 검증 | ⏳ 설계 | — | label·반복관측 데이터 필요 |

> **데이터 제약**: GPS 태깅 주행영상과 전문가/PMS 정답 label이 아직 확보되지 않아 ④~⑥과 검증은 데이터 수집 후 구현한다. 본 제출은 데이터가 있는 ②③③' 까지를 동작 가능한 코드로 구현하고, 나머지는 인터페이스가 호환되도록 설계를 고정한다.

---

## ④ GPS 구간 매핑 (설계)

**목표**: 이미지/프레임 단위 파손 인식 결과를 도로 구간 단위로 귀속.

1. **GPS 좌표 보정** — 주행영상 프레임 timestamp ↔ GPS 로그 동기화, 이상치 제거, 칼만/이동평균 평활.
2. **Map matching** — 보정된 GPS 점열을 도로망(유성구 도시계획시설 도로번호 DWG/DXF → GIS 변환) 링크에 투영. HMM 기반 map matching 또는 최근접 링크 투영.
3. **구간 분할** — 교차로~교차로 또는 일정 길이(예: 100m) 기준으로 구간 정의. 구간 레지스트리는 `유성구 도로.xlsx`(2,805행) 기반.
4. **Spatial join** — 각 파손 인식 결과(프레임 위치)를 포함 구간에 join → `segment_id → [damage records]`.

**입력 인터페이스(예정)**:
```python
# segment_id: str (road_registry.RoadSegment.segment_id)
# 각 프레임 정량화 결과(02_infer_quantify의 per-image row) + (lat, lon)
join_damage_to_segments(per_image_rows, gps_track, road_links) -> dict[str, list[dict]]
```

---

## ⑤ 구간 단위 Feature 구조화 (설계)

구간별 파손 records를 집계하여 feature vector 구성 (발표자료 슬라이드 8):

| Feature | 정의 |
|---|---|
| 균열 점유율 | 구간 내 균열 mask 면적 합 / 구간 노면 면적 |
| 파손 빈도 | 단위 길이당 파손 객체 수 (count / 구간 길이) |
| 심각도 | 클래스별 severity 가중 평균 (포트홀 > 균열) |
| 공간 집중도 | 파손 위치의 공간 분산/군집도 (예: 길이축 히스토그램 엔트로피) |
| 반복 파손 지속도 | 동일·유사 위치 파손의 반복 주행 간 검출 일치도 |
| 보조 | 구간 길이, 주행 횟수, 평균 confidence |

→ 정규화(min-max 또는 z-score) 후 `feature vector` 생성.

---

## ⑥ 포장상태지수(PCI) 산출 + 상태평가 모델 (설계)

### 1차 기준지수 (해석 가능, 가중합)

```
Damage Score = w1·균열점유율 + w2·파손빈도 + w3·심각도
             + w4·공간집중도 + w5·반복파손지속도
Pavement Condition Score = 100 × (1 − Normalized Damage Score)   # 0~100점
```

### 가중치 설정·보정 (발표자료 슬라이드 8)

1. 선행연구 기반 초기 가중치 범위 설정
2. 전문가 **AHP** 쌍대비교로 상대 가중치 산정
3. 전문가/PMS 결과와 **Spearman** 상관 비교로 보정
4. **민감도 분석** — 가중치 변화 시 순위·등급 안정성 확인

### ML 상태평가 모델

- 입력: 구간 단위 feature vector
- 모델: **XGBoost 또는 LightGBM** (비선형 관계·변수 중요도)
- label: 전문가 평가 또는 기존 PMS 결과
- 출력: 포장상태지수 / 상태등급 / 보수 우선순위
- 해석: feature importance / SHAP

---

## 검증 체계 (설계)

| 평가 대상 | 지표 |
|---|---|
| 탐지·분할 성능 | Precision, Recall, F1, mAP50, mask IoU/Dice |
| 상태평가 성능 | Spearman, MAE/RMSE 또는 Accuracy/F1 |
| 반복 안정성 | ICC(반복측정 일관성), CV(변동계수) |
| 가중치 검증 | AHP 일관성비, 상관성 보정, 민감도 분석 |
| 운영성 | 1km당 처리시간, 조사 소요시간·인력 |

목표: 전문가 평가와 Spearman ≥ 0.75, 동일 구간 반복 ICC ≥ 0.80, 1km 주행영상당 5분 이내 처리.
