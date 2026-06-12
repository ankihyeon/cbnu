"""
src — 지능형 도로포장관리시스템 지원 프레임워크 핵심 모듈 패키지 (프로젝트 #2).

본 패키지는 발표자료(#2)에서 제시한 프레임워크 중 **실데이터 보유분**
(YOLO26-seg 파손 탐지·분할, 파손 정량화, 유성구 도로 구간 레지스트리)의
구현을 담는다. GPS 구간 매핑·포장상태지수(PCI) 산출·XGBoost/LightGBM
상태평가·검증 모듈은 `docs/FRAMEWORK_DESIGN.md`에 설계로 정리되어 있다.
"""

from .io_utils import imread_unicode, imwrite_unicode
from .polygon_to_mask import (
    yolo_seg_to_masks, yolo_seg_to_mask,
    CLASS_NAMES, CRACK_CLASSES, POTHOLE_CLASSES, DAMAGE_CLASSES,
)
from .damage_quantify import (
    quantify_image, quantify_class_mask, DEFAULT_SEVERITY,
)
from .road_registry import (
    load_road_registry, summarize, to_records, RoadSegment,
)

__all__ = [
    "imread_unicode", "imwrite_unicode",
    "yolo_seg_to_masks", "yolo_seg_to_mask",
    "CLASS_NAMES", "CRACK_CLASSES", "POTHOLE_CLASSES", "DAMAGE_CLASSES",
    "quantify_image", "quantify_class_mask", "DEFAULT_SEVERITY",
    "load_road_registry", "summarize", "to_records", "RoadSegment",
]

__version__ = "1.0.0"
