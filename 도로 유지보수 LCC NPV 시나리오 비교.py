"""LCC 시나리오 NPV 계산 — 할인율 가변.

비용은 단가표(repair_method_cost) 기반: 단가(₩/m²) × 면적(m²).
시나리오 비용차는 열화곡선에서 도출된 공법·시점(year)으로부터 계산한다
(미루면 더 떨어져 더 비싼 공법 → 더 큰 NPV). 추상 임의값 금지.
"""
from __future__ import annotations


def npv(cashflows: list[float], discount_rate: float) -> float:
    """index=연도(t=0..) 현금흐름 리스트의 순현재가치."""
    return round(sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cashflows)), 4)


def scenario_cost(method_code: str, area_m2: float, cost_table: dict[str, int]) -> float:
    """공법 단가(₩/m²) × 면적(m²). 미등록 공법은 KeyError."""
    return float(cost_table[method_code]) * area_m2


def _single_event_npv(method: str, year: int, area_m2: float,
                      discount_rate: float, cost_table: dict[str, int]) -> float:
    cost = scenario_cost(method, area_m2, cost_table)
    cashflows = [0.0] * year + [cost]
    return npv(cashflows, discount_rate)


def compare_scenarios(area_m2: float, immediate: dict, delayed: dict, neglect: dict,
                      discount_rate: float, cost_table: dict[str, int]) -> dict[str, float]:
    """즉시/지연/방치 3시나리오의 NPV.

    각 시나리오는 {"method": 공법코드, "year": 시행연도}.
    """
    return {
        "immediate_npv": _single_event_npv(
            immediate["method"], immediate["year"], area_m2, discount_rate, cost_table),
        "delayed_npv": _single_event_npv(
            delayed["method"], delayed["year"], area_m2, discount_rate, cost_table),
        "neglect_npv": _single_event_npv(
            neglect["method"], neglect["year"], area_m2, discount_rate, cost_table),
    }
