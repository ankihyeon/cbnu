"""
03_build_road_registry.py
=========================
유성구 도로 구간 마스터(`유성구 도로.xlsx`)를 읽어 표준화된
도로 구간 레지스트리(CSV + 요약 JSON)를 생성 (프레임워크 모듈 ③의 기반).

이 레지스트리가 GPS 구간 매핑 시 spatial join의 대상 구간 목록이 된다.

사용법:
    python scripts/03_build_road_registry.py \
      --xlsx "<유성구 도로.xlsx 경로>" \
      --output results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.road_registry import load_road_registry, summarize, to_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="유성구 도로.xlsx 경로")
    parser.add_argument("--output", default="results")
    parser.add_argument("--sheet", default="0", help="시트 인덱스 또는 이름")
    args = parser.parse_args()

    sheet: int | str
    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 로드: {args.xlsx}")
    segments = load_road_registry(Path(args.xlsx), sheet=sheet)
    print(f"  → {len(segments)}개 도로 구간")

    print("[2/3] CSV 저장")
    records = to_records(segments)
    csv_path = out / "road_registry.csv"
    if records:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    print("[3/3] 요약 JSON 저장")
    report = summarize(segments)
    (out / "road_registry_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] 완료")
    print(f"  - {csv_path}")
    print(f"  - {out / 'road_registry_report.json'}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
