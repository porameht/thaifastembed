from typing import Union, List
from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod

NumpyArray = Union[
    NDArray[np.float64],
    NDArray[np.float32],
    NDArray[np.float16],
    NDArray[np.int8],
    NDArray[np.int64],
    NDArray[np.int32],
]

IntArray = Union[NDArray[np.int64], NDArray[np.int32], NDArray[np.int8]]

class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass