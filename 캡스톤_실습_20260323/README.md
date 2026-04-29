# 3주차 실습 - OpenCV 기반 영상 처리

> 충북대학교 지능화 캡스톤 프로젝트 (2026년 3월 23일 실습)

본 실습은 OpenCV 라이브러리를 활용한 디지털 영상 처리의 기초를 다룹니다.
이미지 입출력부터 색 공간 변환, 히스토그램 분석, 영상 필터링까지
영상 처리의 핵심 개념을 단계적으로 학습합니다.

---

## 1. 개발 환경

| 항목 | 내용 |
|------|------|
| OS | Windows 11 |
| Python | 3.10 이상 권장 |
| 주요 라이브러리 | OpenCV (cv2), NumPy, Matplotlib |
| 실행 환경 | Jupyter Notebook (VS Code) |

### 의존성 설치

```bash
pip install opencv-python numpy matplotlib
```

---

## 2. 디렉토리 구조

```
캡스톤_실습_20260323/
├── 실습파일.ipynb              # 메인 실습 노트북 (9개 예제 포함)
├── README.md                   # 본 설명 파일
│
├── Lenna.png                   # 표준 테스트 이미지 (색공간/히스토그램/필터링)
├── candies.png                 # 색상 검출 실습용 (색깔 캔디 추출)
├── Hawkes.jpg                  # 히스토그램 평활화/스트레칭 실습용
├── desert.JPG                  # 히스토그램 역투영 실습용
├── apple.png, mountain.png     # 추가 테스트 이미지
├── test_video.mp4              # 동영상 입출력 실습용
│
├── Average Filter.jpg          # ex9 출력 결과 - 평균값 필터
├── Sharpening Filter.jpg       # ex9 출력 결과 - 샤프닝 필터
└── Laplacian Filter.jpg        # ex9 출력 결과 - 라플라시안 필터
```

---

## 3. 실습 내용 요약

### ex1. 이미지 입출력
- `cv2.imread()`로 Lenna.png를 읽고 `cv2.imshow()`로 화면에 출력
- **[실습]** 동일 이미지를 그레이스케일(`IMREAD_GRAYSCALE`)로 읽어 출력

### ex2. 동영상 파일 입출력
- `cv2.VideoCapture()`로 test_video.mp4를 프레임 단위로 읽어 재생
- `q` 키 입력 시 종료

### ex3. 카메라 영상 입출력
- 노트북 내장 카메라(ID 0)에서 실시간 영상을 받아 출력
- 카메라가 없으면 동작하지 않음 (예외 처리 포함)

### ex4. 색 분리 (BGR)
- `cv2.split()`로 Lenna 이미지의 Blue, Green, Red 채널을 각각 분리
- 채널별 픽셀 값을 numpy 배열 형태로 출력하고 윈도우에 표시

### ex5. 색 공간 변환 (BGR → HSV)
- `cv2.cvtColor()`로 BGR → HSV 변환 후 H, S, V 성분 분리
- **[실습]** YUV 색공간으로도 변환하여 Y, U, V 성분 출력

### ex6. 색상 영역 검출
- **RGB 기반**: candies.png에서 Red ≥ 50인 영역을 마스크로 추출
- **HSV 기반 (빨강)**: Hue 0–10, 170–180 두 구간을 OR로 합쳐 빨간 캔디 추출
- **[실습] HSV 기반 (파랑)**: Hue 100–140 범위를 사용해 파란 캔디 추출

### ex7. 히스토그램 분석
- **Grayscale**: `cv2.calcHist()`로 명도 히스토그램 계산 후 matplotlib로 시각화
- **BGR**: 채널별로 히스토그램을 계산해 한 그래프에 컬러로 동시 표시

### ex8. 명암 보정
| 기법 | 함수 | 설명 |
|------|------|------|
| 명암비 조절 | `(1+α)·I − α·128` | α 값에 따라 대비를 강조/완화 |
| 평활화 (Equalization) | `cv2.equalizeHist()` | 히스토그램을 균등 분포로 재배치 |
| 스트레칭 (Normalization) | `cv2.normalize(NORM_MINMAX)` | 명도 범위를 0–255로 확장 |
| 역투영 (BackProjection) | `cv2.calcBackProject()` | ROI의 색 분포로 유사 영역 검출 (desert.JPG) |

### ex9. 이미지 필터링
3x3 커널을 `cv2.filter2D()`로 적용하고 결과를 jpg로 저장:

| 필터 | 커널 | 효과 |
|------|------|------|
| 평균값 (Average) | 1/9 × ones(3,3) | 노이즈 완화, 이미지 흐림 |
| 샤프닝 (Sharpening) | 중심 5, 상하좌우 −1 | 경계 강조, 선명도 향상 |
| 라플라시안 (Laplacian) | 중심 −4, 상하좌우 1 | 2차 미분, 에지 검출 |

---

## 4. 실행 방법

### Jupyter Notebook 실행
1. VS Code에서 `실습파일.ipynb` 열기
2. Python 인터프리터 선택 (OpenCV가 설치된 환경)
3. 첫 번째 셀의 작업 디렉토리 경로를 본인 환경에 맞게 수정
   ```python
   os.chdir(r'D:\창고\vscode_AI\github\캡스톤_실습_20260323')
   ```
4. 셀을 위에서부터 순서대로 실행

### 주의 사항
- `cv2.imshow()` 창은 **아무 키나 누르면 닫히도록** 설정되어 있음 (`cv2.waitKey(0)`)
- 동영상/카메라 예제(ex2, ex3)는 **`q` 키로 종료**
- ex8 역투영(backproj)은 마우스로 ROI를 드래그한 뒤 `Enter` 또는 `Space`를 눌러 확정
- VS Code 인라인이 아닌 **별도 OpenCV 윈도우**로 출력되므로 작업표시줄을 확인할 것

---

## 5. 학습 목표

- OpenCV의 기본 입출력 API 숙지 (`imread`, `imshow`, `VideoCapture`)
- BGR / HSV / YUV 등 다양한 색 공간의 특성과 활용 이해
- 마스킹과 비트 연산을 이용한 색상 영역 검출 기법 습득
- 히스토그램을 통한 이미지 통계 분석 및 명암 보정 기법 적용
- 컨볼루션 커널 설계와 공간 필터링의 효과 비교

---

## 6. 참고

- OpenCV 공식 문서: https://docs.opencv.org/
- Lenna 이미지는 영상처리 분야의 표준 벤치마크 이미지입니다.
