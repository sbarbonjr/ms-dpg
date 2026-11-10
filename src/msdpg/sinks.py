"""Compatibility container for legacy sink-extraction consumers."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SinkExtraction:
    """Executed path records and label-specific sink/predicate masses."""

    path_records: list[dict]
    sink_mass: np.ndarray
    predicate_mass: list[dict]
    gains: list[dict]
