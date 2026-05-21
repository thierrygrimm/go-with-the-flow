from .base import GenerativeMethod
from .ddpm import DDPM
from .flow import FlowMatching
from .unet import SIZES, UNet


METHODS: dict[str, type[GenerativeMethod]] = {"ddpm": DDPM, "fm": FlowMatching}


def build(method: str, size: str, shape: tuple[int, int, int]) -> GenerativeMethod:
    unet = UNet.for_size(size, in_channels=shape[0], out_channels=shape[0])
    return METHODS[method](shape=shape, model=unet)
