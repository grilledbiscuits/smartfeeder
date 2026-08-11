"""Augmentation that mimics DEPLOYMENT failure modes, not photography.

The domain gap between clean field-guide photographs and a real feeder camera is
the single biggest technical risk in this project, and until real capture data
exists augmentation is the only lever on it. So the pipeline is built around
what actually goes wrong at a feeder, not around the standard flip-and-crop:

* **Motion blur** -- sunbirds are fast and hover. Directional, not Gaussian:
  a bird crossing frame smears along one axis.
* **Exposure and backlight** -- a feeder against bright sky blows highlights;
  the same feeder at dawn is near-silhouette.
* **Occlusion cutout** -- the bird's head is frequently inside a feeder port,
  so a contiguous chunk of the most informative region is simply absent.
* **JPEG artefacts** -- the capture path re-encodes, and INT8-era artefacts
  interact badly with fine plumage detail.
* **Partial-frame crops** -- birds arrive and leave; half a bird is common.

Every parameter is read from config/train.yaml. None of it is validated against
real feeder footage yet -- see ASSUMPTIONS.md A10.
"""

from __future__ import annotations

import random

import torch
from torchvision.transforms import v2


class DirectionalMotionBlur(torch.nn.Module):
    """Blur along a random axis, approximating a bird crossing the frame.

    torchvision's GaussianBlur is isotropic, which looks like defocus rather
    than motion. A real fast-moving subject smears along its direction of
    travel, so the kernel is a line at a random angle.
    """

    def __init__(self, p: float = 0.35, kernel_range: tuple[int, int] = (3, 11)) -> None:
        super().__init__()
        self.p = p
        self.lo, self.hi = kernel_range

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        k = random.randrange(self.lo | 1, self.hi | 1 + 1, 2)  # odd sizes only
        kernel = torch.zeros(k, k)
        angle = random.uniform(0, 3.14159)
        cx = cy = k // 2
        for i in range(k):
            offset = i - cx
            x = int(round(cx + offset * torch.cos(torch.tensor(angle)).item()))
            y = int(round(cy + offset * torch.sin(torch.tensor(angle)).item()))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        total = kernel.sum()
        if total == 0:
            return img
        kernel = (kernel / total).expand(img.shape[-3], 1, k, k)
        pad = k // 2
        x = torch.nn.functional.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
        return torch.nn.functional.conv2d(x, kernel, groups=img.shape[-3]).squeeze(0)


def build_train_transform(cfg, image_size: int):
    """Training augmentation, driven entirely by config."""
    a = cfg.train_cfg["train"]["augment"]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    steps: list = [
        # Partial-frame crops: scale_jitter's lower bound is what produces
        # "half a bird", which is common as birds arrive and leave.
        v2.RandomResizedCrop(
            image_size, scale=tuple(a["scale_jitter"]), antialias=True
        ),
        v2.RandomHorizontalFlip(p=a["hflip"]),
        v2.RandomRotation(degrees=a["rotate_deg"]),
    ]

    exp = a["exposure_shift"]
    steps.append(
        v2.RandomApply(
            [v2.ColorJitter(brightness=exp["brightness"], contrast=exp["contrast"],
                            saturation=0.2, hue=0.03)],
            p=exp["p"],
        )
    )

    jpg = a["jpeg_artifacts"]
    # v2.JPEG operates on uint8, so it must run before the float conversion.
    steps.append(
        v2.RandomApply([v2.JPEG(quality=tuple(jpg["quality_range"]))], p=jpg["p"])
    )

    steps += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]

    mb = a["motion_blur"]
    steps.append(DirectionalMotionBlur(p=mb["p"], kernel_range=tuple(mb["kernel_range"])))

    steps.append(v2.Normalize(mean=mean, std=std))

    occ = a["occlusion_cutout"]
    # After normalisation, so the erased region is the dataset mean (zero),
    # which is what an absent region should look like to the network.
    steps.append(
        v2.RandomErasing(p=occ["p"], scale=(0.05, occ["max_area_frac"]), value=0.0)
    )
    return v2.Compose(steps)


def build_eval_transform(image_size: int):
    """Deterministic evaluation transform. No augmentation, ever."""
    return v2.Compose(
        [
            v2.Resize(int(image_size * 1.14), antialias=True),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
