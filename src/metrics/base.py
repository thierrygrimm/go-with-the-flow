from abc import ABC, abstractmethod

from torch import Tensor


class Metric(ABC):
    @abstractmethod
    def update(self, images: Tensor, *, real: bool) -> None: ...

    @abstractmethod
    def compute(self) -> float: ...

    @abstractmethod
    def reset(self) -> None: ...
