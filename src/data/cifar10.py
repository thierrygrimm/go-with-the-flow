import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from .base import DatasetBuilder


class CIFAR10(DatasetBuilder):
    def __init__(self, root: str = "./datasets", num_workers: int = 2):
        self.root = root
        self.num_workers = num_workers

    @property
    def shape(self) -> tuple[int, int, int]:
        return 3, 32, 32

    @staticmethod
    def _transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def train_loader(self, batch_size: int) -> DataLoader:
        ds = torchvision.datasets.CIFAR10(
            self.root, train=True, download=True, transform=self._transform(),
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True, persistent_workers=self.num_workers > 0,
        )

    def eval_loader(self, batch_size: int) -> DataLoader:
        ds = torchvision.datasets.CIFAR10(
            self.root, train=False, download=True, transform=self._transform(),
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True, persistent_workers=self.num_workers > 0,
        )
