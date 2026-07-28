"""열화곡선(Deterioration Curve) 계산 — 크랙율(%) 입력 기반.

KICT 이상혁 박사 방법론: 상태 척도가 아니라 크랙율(단위면적당 %)을 y축으로 사용.
PCI/NHPCI 산식에 의존하지 않는다(설계 스펙 risk #4). 골격은 선형 외삽(고정 기울기).
"""
from __future__ import annotations

# 도로 등급별 표준 곡선 (러프 골격 — 이력 0건 fallback)
# initial_crack_rate: 신설 직후 크랙율(%), annual_increase: 연간 크랙율 증가(%/년)
STANDARD_CURVES: dict[str, dict[str, float]] = {
    "중로": {"initial_crack_rate": 2.0, "annual_increase": 2.5},
    "소로": {"initial_crack_rate": 3.0, "annual_increase": 3.5},
    "default": {"initial_crack_rate": 2.5, "annual_increase": 3.0},
}


def assign_standard_curve(road_grade: str | None) -> dict[str, float]:
    """도로 등급 문자열 → 표준 곡선 파라미터. 미지의 등급은 default."""
    return STANDARD_CURVES.get(road_grade or "default", STANDARD_CURVES["default"])


def project_crack_rate(curve: dict[str, float], current_crack_rate: float, years: float) -> float:
    """현재 크랙율에서 years년 후 크랙율(선형 외삽, 0~100 clamp)."""
    projected = current_crack_rate + curve["annual_increase"] * years
    return max(0.0, min(100.0, projected))


def years_to_trigger(curve: dict[str, float], current_crack_rate: float,
                     trigger_crack_rate: float) -> float:
    """현재 크랙율 → 트리거 크랙율 도달까지 연수. 이미 초과면 0."""
    if current_crack_rate >= trigger_crack_rate:
        return 0.0
    annual_increase = curve.get("annual_increase", 0.0)
    if not annual_increase:  # 0 또는 키 누락 → 0으로 나눗셈 방어 (표준 곡선은 전부 nonzero)
        return 0.0
    gap = trigger_crack_rate - current_crack_rate
    return round(gap / annual_increase, 4)


def build_curve_series(curve: dict[str, float], current_crack_rate: float,
                       horizon_years: int,
                       reset_years: list[int] | None = None) -> list[tuple[int, float]]:
    """0..horizon_years 각 연도의 (year, crack_rate) 시계열.

    reset_years 에 포함된 연도에는 진짜 유지보수가 일어난 것으로 보고
    크랙율을 곡선 initial 로 리셋한다(KICT: 덧씌우기/절삭만 곡선 리셋)."""
    reset_set = set(reset_years or [])
    series: list[tuple[int, float]] = []
    crack = current_crack_rate
    for year in range(horizon_years + 1):
        if year in reset_set:
            crack = curve["initial_crack_rate"]
        series.append((year, round(crack, 4)))
        crack = max(0.0, min(100.0, crack + curve["annual_increase"]))
    return series
