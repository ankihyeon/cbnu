"""
road_registry.py
================
유성구 도로 구간 마스터(`유성구 도로.xlsx`) 로더.

본 프레임워크의 분석 단위는 "도로 구간"(교차로~교차로 또는 일정 길이)이며,
유성구청이 제공하는 도로 대장이 그 구간 레지스트리 역할을 한다.
GPS 구간 매핑(설계 단계) 시, 주행영상의 파손 인식 결과를 이 레지스트리의
각 구간에 spatial join 하여 구간 단위 Feature를 구성한다.

원본 xlsx 구조 (헤더가 3번째 행에 위치):
    NO | 시설물명 | 업종명 | 도로명주소 | 관리부서명 | 도로시작위치 | 도로종료위치 | 기준값 | 기준값단위
    └ 시설물명 = 도로번호, 기준값(km) = 구간 연장
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


# 원본 헤더(한글) → 표준 필드명
COLUMN_MAP = {
    "NO": "no",
    "시설물명": "road_no",
    "업종명": "category",
    "도로명주소": "road_address",
    "관리부서명": "dept",
    "도로시작위치": "start_loc",
    "도로종료위치": "end_loc",
    "기준값": "length_km",
    "기준값단위": "length_unit",
}

# 헤더 행 위치 (0-based). 원본은 1~2행이 비어 있고 3행이 헤더.
HEADER_ROW = 2


@dataclass(frozen=True)
class RoadSegment:
    """도로 구간 1건 (불변)."""
    no: str
    road_no: str
    category: str
    road_address: str
    dept: str
    start_loc: str
    end_loc: str
    length_km: float
    length_unit: str

    @property
    def segment_id(self) -> str:
        """구간 고유 ID (도로번호 기반, 없으면 NO 사용)."""
        return f"R{self.road_no}" if self.road_no else f"NO{self.no}"


def _to_float(v) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except (ValueError, AttributeError, TypeError):
        return 0.0


def load_road_registry(xlsx_path: Path,
                       sheet: int | str = 0,
                       header_row: int = HEADER_ROW) -> list[RoadSegment]:
    """
    유성구 도로 xlsx → RoadSegment 리스트.

    Args:
        xlsx_path: `유성구 도로.xlsx` 경로
        sheet: 시트 인덱스/이름
        header_row: 헤더 행(0-based). 기본 2 (3번째 행).

    Returns:
        RoadSegment 리스트
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=header_row,
                       dtype=str, engine="openpyxl")
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    segments: list[RoadSegment] = []
    for _, row in df.iterrows():
        no = str(row.get("no", "") or "").strip()
        road_no = str(row.get("road_no", "") or "").strip()
        # 완전 빈 행 skip
        if not no and not road_no:
            continue
        segments.append(RoadSegment(
            no=no,
            road_no=road_no,
            category=str(row.get("category", "") or "").strip(),
            road_address=str(row.get("road_address", "") or "").strip(),
            dept=str(row.get("dept", "") or "").strip(),
            start_loc=str(row.get("start_loc", "") or "").strip(),
            end_loc=str(row.get("end_loc", "") or "").strip(),
            length_km=_to_float(row.get("length_km", 0)),
            length_unit=str(row.get("length_unit", "") or "").strip(),
        ))
    return segments


# 길이 단위 → km 환산 계수. 원본은 "km"/"㎞"(U+339E)/"m"가 혼재.
# 그 외(㎡·대·개 등 면적·개수 단위)는 길이 집계에서 제외.
_LENGTH_TO_KM = {"km": 1.0, "㎞": 1.0, "m": 0.001, "ｋｍ": 1.0}


def normalized_length_km(seg: RoadSegment) -> float | None:
    """길이 단위면 km로 환산, 길이 단위가 아니면 None."""
    factor = _LENGTH_TO_KM.get(seg.length_unit)
    return seg.length_km * factor if factor is not None else None


def summarize(segments: list[RoadSegment]) -> dict:
    """
    레지스트리 통계 요약.

    원본 xlsx의 기준값 단위가 km/m/개소/필지 등으로 혼재되어 있어,
    길이 집계는 km·m(길이 단위)만 km로 환산하여 산출하고
    기타 단위 행은 별도로 카운트한다.
    """
    from collections import Counter

    by_category = Counter(s.category for s in segments)
    by_dept = Counter(s.dept for s in segments)
    by_unit = Counter(s.length_unit or "(없음)" for s in segments)

    length_km = [v for v in (normalized_length_km(s) for s in segments)
                 if v is not None and v > 0]
    non_length_rows = sum(1 for s in segments
                          if normalized_length_km(s) is None)
    total_km = sum(length_km)

    return {
        "segment_count": len(segments),
        "length_unit_rows": non_length_rows,            # 길이 단위가 아닌 행 수
        "length_segment_count": len(length_km),         # 길이 집계에 포함된 구간 수
        "total_length_km": round(total_km, 3),
        "mean_length_km": round(total_km / len(length_km), 4) if length_km else 0.0,
        "min_length_km": round(min(length_km), 3) if length_km else 0.0,
        "max_length_km": round(max(length_km), 3) if length_km else 0.0,
        "by_unit": dict(by_unit.most_common()),
        "by_category": dict(by_category.most_common()),
        "by_dept": dict(by_dept.most_common(10)),
    }


def to_records(segments: list[RoadSegment]) -> list[dict]:
    """직렬화용 dict 리스트 (segment_id 포함)."""
    out = []
    for s in segments:
        d = asdict(s)
        d["segment_id"] = s.segment_id
        out.append(d)
    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="유성구 도로.xlsx 경로")
    args = parser.parse_args()

    segs = load_road_registry(Path(args.xlsx))
    print(json.dumps(summarize(segs), ensure_ascii=False, indent=2))
