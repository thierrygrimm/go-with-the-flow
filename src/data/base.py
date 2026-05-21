from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class DatasetBuilder(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]: ...

    @abstractmethod
    def train_loader(self, batch_size: int) -> DataLoader: ...

    @abstractmethod
    def eval_loader(self, batch_size: int) -> DataLoader: ...
