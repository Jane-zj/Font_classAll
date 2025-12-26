"""
自定义图像变换（满足新规则）：
1) 若原图 w < 600 且 h < 100，则居中 padding 到画布 600x100；
2) 否则，先按照 6:1 进行居中 padding（画布尺寸不小于原图），再等比缩放到 600x100（最终尺寸 H=100, W=600）。

使用方式：
from transforms import PadToAspectOrCanvas, build_transforms

train_tfms, val_tfms = build_transforms()
"""

from typing import Tuple
import math
from PIL import Image, ImageOps
import torchvision.transforms as T


class PadToAspectOrCanvas:
    """
    规则：
    - 若原图 w < small_w 且 h < small_h：直接居中 padding 到画布 (small_w, small_h)。
    - 否则：按照 aspect_ratio（默认 6:1）居中 padding（画布尺寸不小于原图），随后由上层进行 Resize 到 (small_h, small_w)。

    注：填充颜色默认黑色 (0,0,0)，如需白底可传 fill=(255,255,255)。
    """

    def __init__(self, small_w: int = 600, small_h: int = 100, aspect_ratio: float = 6.0, fill: Tuple[int, int, int] = (0, 0, 0)):
        assert aspect_ratio > 0, "aspect_ratio 必须为正值"
        self.small_w = int(small_w)
        self.small_h = int(small_h)
        self.aspect_ratio = float(aspect_ratio)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size  # PIL: (width, height)

        if w < self.small_w and h < self.small_h:
            # 直接 pad 到 600x100
            w_pad, h_pad = self.small_w, self.small_h
        else:
            # 先按 6:1 进行 padding，画布不小于原图
            h_pad = max(h, int(math.ceil(w / self.aspect_ratio)))
            w_pad = int(math.ceil(h_pad * self.aspect_ratio))

        pad_left = (w_pad - w) // 2
        pad_right = w_pad - w - pad_left
        pad_top = (h_pad - h) // 2
        pad_bottom = h_pad - h - pad_top

        img_padded = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return img_padded


def build_transforms(
    pad_fill: Tuple[int, int, int] = (0, 0, 0),
    image_size_hw: Tuple[int, int] = (100, 600),
    use_imagenet_norm: bool = True,
):
    """
    构建训练/验证时的图像变换流水线，遵循新规则并最终输出到 (H=100, W=600)。

    参数：
    - pad_fill: padding 颜色，默认黑色；可改为 (255,255,255) 白色。
    - image_size_hw: 最终尺寸 (H, W)，默认 (100, 600)。
    - use_imagenet_norm: 是否使用 ImageNet 的均值方差归一化（若加载预训练权重，建议 True）。
    """

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if use_imagenet_norm else T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transforms = T.Compose([
        PadToAspectOrCanvas(small_w=600, small_h=100, aspect_ratio=6.0, fill=pad_fill),
        T.Resize(image_size_hw),  # PIL: (H, W) -> (100, 600)
        # 轻量数据增强（可按需调整或关闭）
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
        normalize,
    ])

    val_transforms = T.Compose([
        PadToAspectOrCanvas(small_w=600, small_h=100, aspect_ratio=6.0, fill=pad_fill),
        T.Resize(image_size_hw),
        T.ToTensor(),
        normalize,
    ])

    return train_transforms, val_transforms