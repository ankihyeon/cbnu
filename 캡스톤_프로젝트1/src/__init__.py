"""
src — 알고리즘 핵심 모듈 패키지.

본 패키지는 Li et al. (2024) 논문의 BG / Optimized OrthoBoundary 알고리즘
구현 및 보조 유틸리티를 포함한다.
"""

from .bg_algorithm import compute_crack_length
from .orthoboundary import compute_crack_widths, compute_crack_widths_parallel
from .polygon_to_mask import yolo_seg_to_mask, batch_convert
from .io_utils import imread_unicode, imwrite_unicode

__all__ = [
    "compute_crack_length",
    "compute_crack_widths",
    "compute_crack_widths_parallel",
    "yolo_seg_to_mask",
    "batch_convert",
    "imread_unicode",
    "imwrite_unicode",
]

__version__ = "1.0.0"
