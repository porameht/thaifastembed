"""Sparse embedding representation for BM25 vectors."""

from dataclasses import dataclass
from .types import NumpyArray, IntArray
import numpy as np


@dataclass
class SparseEmbedding:
    values: NumpyArray
    indices: IntArray

    def as_object(self) -> dict[str, NumpyArray]:
        return {
            "values": self.values,
            "indices": self.indices,
        }

    def as_dict(self) -> dict[int, float]:
        return {int(i): float(v) for i, v in zip(self.indices, self.values)}

    @classmethod
    def from_dict(cls, data: dict[int, float]) -> "SparseEmbedding":
        if len(data) == 0:
            return cls(values=np.array([]), indices=np.array([]))
        indices, values = zip(*data.items())
        return cls(values=np.array(values), indices=np.array(indices))
